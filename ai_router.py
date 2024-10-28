import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai.error import OpenAIError

from modal import Image, App, asgi_app, Secret

# Configure logging
logging.basicConfig(level=logging.INFO)

web_app = FastAPI(
    title="LLM Comparison API",
    description="API for interacting with different language models.",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "https://eval.supa.so",
]

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image = Image.debian_slim().pip_install(
    ["litellm", "supabase", "pydantic==2.5.3", "fastapi==0.109.0", "openai"]
)
llm_compare_app = App(
    name="llm-compare-api",
    image=image,
    secrets=[
        Secret.from_name("SUPABASE_SECRETS"),
        Secret.from_name("OLLAMA_API"),
    ],
)

with llm_compare_app.image.imports():
    from litellm import completion
    from supabase import create_client, Client

    # Initialize Supabase client
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)


# Pydantic models
class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None


class Message(BaseModel):
    content: str = Field(..., description="Content of the message.")
    role: str = Field(..., description="Role of the sender.")
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None
    logprobs: Optional[dict] = None


class Choice(BaseModel):
    finish_reason: Optional[str] = Field(
        None, description="Reason the generation stopped."
    )
    index: int = Field(..., description="Index of the choice.")
    message: Message = Field(..., description="The message content.")


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[dict] = None
    prompt_tokens_details: Optional[dict] = None


class ModelResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: float
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[Choice]
    usage: Usage


class MessageRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use.")
    message: str = Field(..., description="Message text to send to the model.")
    api_key: Optional[str] = Field(
        None, description="API key if required by the provider."
    )


class ModelInfo(BaseModel):
    provider: str = Field(..., description="Provider of the model.")
    model_name: str = Field(..., description="Name of the model.")


async def fetch_models_from_supabase() -> List[dict]:
    logging.info("Fetching models from Supabase")
    response = supabase.table("available_models").select("*").execute()
    logging.info(f"Fetched {len(response.data)} models")
    return response.data


@web_app.post(
    "/message",
    response_model=ModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Send message to a model",
    responses={
        200: {"description": "Successful response with the model's reply."},
        400: {"description": "Bad Request. Model or message not provided."},
        404: {"description": "Model not supported."},
        500: {"description": "Internal Server Error."},
    },
)
async def messaging(request: MessageRequest):
    logging.info("Received /message request")
    model_name = request.model
    message = request.message
    api_key = request.api_key
    logging.info(f"Model name: {model_name}")
    logging.info(f"Message: {message}")

    # Fetch models from Supabase
    models = await fetch_models_from_supabase()

    # Check if the model is supported
    model_info = next((m for m in models if m["model_name"] == model_name), None)
    if not model_info:
        logging.warning(f"Model {model_name} not supported")
        raise HTTPException(status_code=404, detail="Model not supported")

    provider = model_info["provider"]
    logging.info(f"Provider: {provider}")

    try:
        if provider == "ollama":
            model_name = f"ollama/{model_name}"
            logging.info(f"Updated model name for Ollama: {model_name}")
            api_url = os.environ["OLLAMA_API_URL"]
            logging.info(f"API URL for Ollama: {api_url}")
            try:
                response_obj = completion(
                    model=model_name,
                    messages=[{"content": message, "role": "user"}],
                    api_base=api_url,
                )
            except OpenAIError as e:
                logging.error(f"Error during Ollama completion: {e}")
                raise HTTPException(status_code=500, detail="Error during Ollama completion")
        elif provider == "github":
            model_name = f"github/{model_name}"
            logging.info(f"Updated model name for GitHub: {model_name}")
            try:
                response_obj = completion(
                    model=model_name,
                    messages=[{"content": message, "role": "user"}],
                )
            except OpenAIError as e:
                logging.error(f"Error during GitHub completion: {e}")
                raise HTTPException(status_code=500, detail="Error during GitHub completion")
        elif provider in ["openai", "anthropic"]:
            if not api_key:
                logging.info("No API key provided, defaulting to GitHub provider")
                model_name = f"github/{model_name}"
                logging.info(f"Updated model name for GitHub: {model_name}")
                try:
                    response_obj = completion(
                        model=model_name,
                        messages=[{"content": message, "role": "user"}],
                    )
                except OpenAIError as e:
                    logging.error(f"Error during GitHub completion: {e}")
                    raise HTTPException(status_code=500, detail="Error during GitHub completion")
            else:
                os.environ["API_KEY"] = api_key
                logging.info("Using API key for provider")
                try:
                    response_obj = completion(
                        model=model_name,
                        messages=[{"content": message, "role": "user"}],
                    )
                except OpenAIError as e:
                    logging.error(f"Error during {provider} completion: {e}")
                    raise HTTPException(status_code=500, detail=f"Error during {provider} completion")
        else:
            logging.warning(f"Provider {provider} is not supported")
            raise HTTPException(status_code=400, detail="Provider not supported")
    finally:
        if provider in ["openai", "anthropic"] and api_key:
            del os.environ["API_KEY"]

    return response_obj


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
    logging.info("Received /list_models request")
    models = await fetch_models_from_supabase()
    logging.info(f"Returning {len(models)} models")
    return models


@llm_compare_app.function()
@asgi_app()
def fastapi_app():
    logging.info("Starting FastAPI app")
    return web_app
