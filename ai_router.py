import os
import httpx
import logging
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from modal import Image, App, asgi_app, Secret

# Configure logging
logging.basicConfig(level=logging.INFO)

web_app = FastAPI(
    title="LLM Comparison API",
    description="API for interacting with different language models.",
    version="1.0.0",
)

image = Image.debian_slim().pip_install(
    ["litellm", "supabase", "pydantic==2.5.3", "fastapi==0.109.0"]
)
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


# Pydantic models for request and response
class MessageRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use.")
    messages: List[dict] = Field(
        ..., description="List of message objects to send to the model."
    )


class MessageResponse(BaseModel):
    response: str = Field(..., description="The model's response to the messages.")


class ModelInfo(BaseModel):
    provider: str = Field(..., description="Provider of the model.")
    model: str = Field(..., description="Name of the model.")


async def fetch_models_from_supabase() -> List[dict]:
    """
    Fetch the list of available models from Supabase.

    Returns:
        A list of models with their provider and model name.

    Raises:
        HTTPException: If an error occurs while fetching models.
    """
    logging.info("Fetching models from Supabase")
    response = supabase.table("available_models").select("*").execute()
    if response.error:
        logging.error(f"Error fetching models: {response.error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching models from Supabase",
        )
    logging.info(f"Fetched {len(response.data)} models")
    return response.data


@web_app.post(
    "/message",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send messages to a model",
    responses={
        200: {"description": "Successful response with the model's reply."},
        400: {"description": "Bad Request. Model or messages not provided."},
        404: {"description": "Model not supported."},
        500: {"description": "Internal Server Error."},
    },
)
async def messaging(request: MessageRequest):
    """
    Send a message to the specified model and receive a response.

    Args:
        request (MessageRequest): The request body containing model name and messages.

    Returns:
        MessageResponse: The response from the model.

    Raises:
        HTTPException:
            - 400 Bad Request: If model or messages are not provided.
            - 404 Not Found: If the specified model is not supported.
            - 500 Internal Server Error: If an error occurs during model completion.
    """
    logging.info("Received /message request")
    model_name = request.model
    messages = request.messages
    logging.info(f"Model name: {model_name}")
    logging.info(f"Messages: {messages}")

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
        response_text = completion(
            model=model_name,
            messages=messages,
            stream=True,
        )
        logging.info("Received response from completion")
    except Exception as e:
        logging.error(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail="Error during model completion")

    return MessageResponse(response=response_text)


@web_app.get(
    "/list_models",
    response_model=List[ModelInfo],
    status_code=status.HTTP_200_OK,
    summary="Get list of available models",
    responses={
        200: {"description": "Successful response with the list of models."},
        500: {"description": "Internal Server Error."},
    },
)
async def list_models():
    """
    Retrieve a list of available models.

    Returns:
        List[ModelInfo]: A list of models with their provider and model name.

    Raises:
        HTTPException: If an error occurs while fetching models.
    """
    logging.info("Received /list_models request")
    models = await fetch_models_from_supabase()
    logging.info(f"Returning {len(models)} models")
    return models


@llm_compare_app.function()
@asgi_app()
def fastapi_app():
    """
    The FastAPI application function for deployment.

    Returns:
        FastAPI: The FastAPI application instance.
    """
    logging.info("Starting FastAPI app")
    return web_app
