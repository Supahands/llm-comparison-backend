import sys
import time
import os
import modal
import subprocess
import logging

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from modal import gpu, Secret, Image

# Configure logging
logging.basicConfig(level=logging.INFO)

# Default server port.
MODEL_IDS: list[str] = [
    "llama3",
    "llama3.2",
    "mistral",
    "gemma2",
    "qwen2.5",
    "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct",
]

OLLAMA_PORT: int = 11434
OLLAMA_URL: str = f"http://localhost:{OLLAMA_PORT}"


def _run_subprocess(cmd: list[str], block: bool = True) -> None:
    if block:
        subprocess.run(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    else:
        subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


def _is_server_healthy() -> bool:
    try:
        response = requests.get(OLLAMA_URL)
        if response.ok:
            print(f"ollama server running => {OLLAMA_URL}")
            return True
        else:
            print(f"ollama server not running => {OLLAMA_URL}")
            return False
    except requests.RequestException as e:
        return False


def download_model():
    _run_subprocess(["ollama", "serve"], block=False)
    while not _is_server_healthy():
        print("waiting for server to start ...")
        time.sleep(1)

    for model in MODEL_IDS:
        # Download all models
        _run_subprocess(["ollama", "pull", model])


def update_model_db():
    print("Updating model database...")
    from supabase import create_client, Client

    # Initialize Supabase client
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)

    # Fetch existing models from the database with provider 'ollama'
    response = (
        supabase.table("available_models")
        .select("*")
        .eq("provider", "ollama")
        .execute()
    )

    existing_models = response.data

    existing_model_names = {model["model_name"] for model in existing_models}
    model_ids_set = set(MODEL_IDS)

    # Models to add
    models_to_add = model_ids_set - existing_model_names
    # Models to remove
    models_to_remove = existing_model_names - model_ids_set

    # Add new models
    for model_name in models_to_add:
        data = {"provider": "ollama", "model_name": model_name}
        print(f"Adding model to DB: {data}")
        insert_response = supabase.table("available_models").insert(data).execute()
        logging.info(f"Added model to DB: {insert_response.data}")

    # Remove outdated models
    for model_name in models_to_remove:
        print(f"Removing model from DB: {model_name}")
        delete_response = (
            supabase.table("available_models")
            .delete()
            .eq("provider", "ollama")
            .eq("model_name", model_name)
            .execute()
        )
        logging.info(f"Removed model from DB: {delete_response.data}")


image = (
    Image.from_registry(
        "ollama/ollama:latest",
        add_python="3.11",
    )
    .pip_install("requests")  # for healthchecks
    .pip_install("httpx")  # for reverse proxy
    .pip_install("supabase")  # Supabase client
    .pip_install("pydantic==2.5.3")
    .pip_install(
        "fastapi==0.109.0"
    )  # Set specific versions, as supabase requires pydantic >=2.5.0
    .copy_local_file("./entrypoint.sh", "/opt/entrypoint.sh")
    .dockerfile_commands(
        [
            "RUN chmod a+x /opt/entrypoint.sh",
            'ENTRYPOINT ["/opt/entrypoint.sh"]',
        ]
    )
    .run_function(download_model)
)

ollama_app = modal.App(
    "ollama-service",
    image=image,
    secrets=[
        Secret.from_name(
            "SUPABASE_SECRETS"
        )  # Ensure this secret contains SUPABASE_URL and SUPABASE_KEY
    ],
)
with ollama_app.image.imports():
    import httpx
    import requests

    from starlette.background import BackgroundTask

    # Start Ollama server and make sure it is running before accepting inputs.
    _run_subprocess(["ollama", "serve"], block=False)
    while not _is_server_healthy():
        print("waiting for server to start ...")
        time.sleep(1)

    print("ollama server started!")

    update_model_db()


# FastAPI proxy. This allows for requests to be handled by Modal, allowing
# effective scaling, queues, etc.
app = FastAPI()


@ollama_app.function(
    gpu=gpu.A10G(count=2), allow_concurrent_inputs=10, concurrency_limit=1
)
@modal.asgi_app()
def ollama_api():
    return app


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
)
async def proxy(request: Request, path: str):
    # Create a new client on every request.
    client = httpx.AsyncClient(
        base_url=OLLAMA_URL, headers=request.headers, timeout=60
    )  # ollama can take a few seconds to respond

    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))

    # Supports both streaming and non-streaming responses.
    async def _response():
        # Handle all other non-streaming methods.
        response = await client.request(
            request.method,
            url,
            params=request.query_params,
            json=await request.json() if len(await request.body()) > 0 else None,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    async def _streaming_response():
        rp_req = client.build_request(
            request.method,
            url,
            content=request.stream(),
            json=await request.json() if len(await request.body()) > 0 else None,
        )

        rp_resp = await client.send(rp_req, stream=True)
        background_task = BackgroundTask(rp_resp.aclose)
        return StreamingResponse(
            rp_resp.aiter_raw(),
            media_type="text/event-stream",
            background=background_task,
        )

    # These two methods support streaming responses.
    # Reference: https://github.com/jmorganca/ollama/blob/main/docs/api.md
    if request.url.path in ("/api/generate", "/api/chat"):
        return await _streaming_response()
    else:
        return await _response()
