import sys
import time
import os
import modal
import subprocess
import logging

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from modal import gpu, Secret, Image
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)

# Default server port.
MODEL_IDS: list[str] = [
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.2:1b",
    "llama3.3",
    "mistral",
    "gemma2",
    "qwen2.5",
    "yi",
    "codellama:7b",
    "codellama:70b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:32b",
    "medllama2",
    "meditron:7b",
    "meditron:70b",
    "mathstral:7b",
    "athene-v2:72b",
    "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct",
    "hf.co/Supa-AI/llama3-8b-cpt-sahabatai-v1-instruct-gguf:Q8_0",
    "hf.co/Supa-AI/llama3-8b-cpt-sahabatai-v1-instruct-gguf:Q2_K",
    "hf.co/Supa-AI/Ministral-8B-Instruct-2410-gguf:Q8_0",
    "hf.co/Supa-AI/gemma2-9b-cpt-sahabatai-v1-instruct-q8_0-gguf",
    "hf.co/Supa-AI/Mixtral-8x7B-Instruct-v0.1-gguf:Q8_0",
    "hf.co/Supa-AI/malaysian-Llama-3.2-3B-Instruct-gguf:Q8_0"
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

    existing_model_ids = {model["model_id"] for model in existing_models}
    model_ids_set = set(MODEL_IDS)

    # Models to add
    models_to_add = model_ids_set - existing_model_ids
    # Models to remove
    models_to_remove = existing_model_ids - model_ids_set

    # Add new models
    for model_name in models_to_add:
        data = {"provider": "ollama", "model_id": model_name, "model_name": model_name}
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
            .eq("model_id", model_name)
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
        "fastapi==0.115.0"
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


class OllamaClient:
    _instance = None
    
    def __init__(self):
        self._client = None
    
    @property
    async def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=OLLAMA_URL,
                timeout=httpx.Timeout(180.0, read=180.0),
                limits=httpx.Limits(max_keepalive_connections=50)
            )
        return self._client
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

ollama_client = OllamaClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Nothing to initialize since client is created lazily
    yield
    # Shutdown: Clean up client
    await ollama_client.close()

# FastAPI proxy. This allows for requests to be handled by Modal, allowing
# effective scaling, queues, etc.
app = FastAPI(lifespan=lifespan)



@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    try:
        client = await ollama_client.client
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))

        async def _streaming_response():
            logging.info(f"Using streaming response for path: {request.url.path}")
            try:
                body = await request.body()
                rp_req = client.build_request(
                    request.method,
                    url,
                    content=request.stream() if not body else None,
                    json=await request.json() if body else None,
                )
                
                rp_resp = await client.send(rp_req, stream=True)
                return StreamingResponse(
                    rp_resp.aiter_raw(),
                    status_code=rp_resp.status_code,
                    media_type="text/event-stream",
                    background=BackgroundTask(rp_resp.aclose)
                )
            except (httpx.ReadError, httpx.ReadTimeout) as e:
                logging.error(f"Streaming error: {str(e)}")
                return Response(
                    content={"error": f"Request failed: {str(e)}"},
                    status_code=504,
                )

        async def _response():
            logging.info(f"Using normal response for path: {request.url.path}")
            body = await request.body()
            response = await client.request(
                request.method,
                url,
                params=request.query_params,
                json=await request.json() if body else None
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )

        if request.url.path in ("/api/generate", "/api/chat"):
            return await _streaming_response()
        return await _response()

    except Exception as e:
        logging.error(f"Proxy error: {str(e)}")
        return Response(
            content={"error": str(e)},
            status_code=500,
            media_type="application/json"
        )

@ollama_app.function(
    gpu=gpu.A10G(count=2), allow_concurrent_inputs=10, concurrency_limit=1, container_idle_timeout=1200,
)
@modal.asgi_app()
def ollama_api():
    return app
