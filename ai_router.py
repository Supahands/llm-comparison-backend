import os
import httpx
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

from modal import Image, App, asgi_app, Secret

# Configure logging
logging.basicConfig(level=logging.INFO)

web_app = FastAPI()

image = Image.debian_slim().pip_install([
    "litellm",
    "supabase",
    "pydantic==2.5.3",
    "fastapi==0.109.0"
])
llm_compare_app = App(
    name="llm-compare-api", image=image, secrets=[Secret.from_name("SUPABASE_SECRETS")]
)

with llm_compare_app.image.imports():
    from litellm import completion
    from supabase import create_client, Client

    # Initialize Supabase client
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)

async def fetch_models_from_supabase():
    logging.info("Fetching models from Supabase")
    response = supabase.table("available_models").select("*").execute()
    logging.info(f"Fetched {len(response.data)} models")
    return response.data

@web_app.post("/message")
async def messaging(request: Request):
    logging.info("Received /message request")
    body = await request.json()
    model_name = body.get("model")
    messages = body.get("messages")
    logging.info(f"Model name: {model_name}")
    logging.info(f"Messages: {messages}")

    if not model_name or not messages:
        logging.warning("Model or messages not provided")
        raise HTTPException(status_code=400, detail="Model and messages are required")

    # Fetch models from Supabase
    models = await fetch_models_from_supabase()

    # Check if the model is supported
    model_info = next((m for m in models if m["model"] == model_name), None)
    if not model_info:
        logging.warning(f"Model {model_name} not supported")
        raise HTTPException(status_code=404, detail="Model not supported")

    provider = model_info["provider"]
    logging.info(f"Provider: {provider}")

    if provider == "ollama":
        model_name = f"ollama/{model_name}"
        logging.info(f"Updated model name for ollama: {model_name}")

    try:
        logging.info(f"Calling completion for model {model_name}")
        response = completion(
            model=model_name,
            messages=messages,
            stream=True,
        )
        logging.info("Received response from completion")
    except Exception as e:
        logging.error(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail="Error during model completion")

    return {"response": response}

@web_app.get("/list_models")
async def list_models():
    logging.info("Received /list_models request")
    models = await fetch_models_from_supabase()
    logging.info(f"Returning {len(models)} models")
    return models

@llm_compare_app.function()
@asgi_app()
def fastapi_app():
    logging.info("Starting FastAPI app")
    return web_app
