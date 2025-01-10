import os
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import contextmanager
from modal import Image, App, asgi_app, Secret, gpu
from const import LIST_OF_REDACTED_WORDS

# Configure logging
logging.basicConfig(level=logging.INFO)

os.environ['LITELLM_LOG'] = 'DEBUG'

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
    [
        "litellm", 
        "supabase", 
        "pydantic==2.5.3", 
        "fastapi==0.109.0", 
        "openai", 
        "langfuse"
    ]
)
llm_compare_app = App(
    name="llm-compare-api",
    image=image,
    secrets=[
        Secret.from_name("SUPABASE_SECRETS"),
        Secret.from_name("OLLAMA_API"),
        Secret.from_name("llm_comparison_github"),
        Secret.from_name("my-huggingface-secret"),
        Secret.from_name("Langfuse-Secret")
    ],
)

with llm_compare_app.image.imports():
    import litellm
    from litellm import completion
    from supabase import create_client, Client
    from openai import OpenAIError
    import re

    litellm.set_verbose=True # 👈 this is the 1-line change you need to make
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"] # logs errors to langfuse

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
    completion_tokens_details: Optional[dict] = Field(default=None, description="Details about completion tokens")
    prompt_tokens_details: Optional[dict] = Field(default=None, description="Details about prompt tokens") 
    response_time: float

    @staticmethod
    def _to_dict(obj):
        """Safely convert an object to a dictionary."""
        if obj is None:
            return None
        try:
            return vars(obj)
        except TypeError:
            # If object doesn't have __dict__, try to convert it to dict directly
            try:
                return dict(obj)
            except (TypeError, ValueError):
                # If conversion fails, return None instead of failing
                return None

    @classmethod
    def from_response(cls, response_obj):
        # Safely convert wrapper objects to dictionaries
        completion_details = cls._to_dict(
            getattr(response_obj.usage, "completion_tokens_details", None)
        )
        prompt_details = cls._to_dict(
            getattr(response_obj.usage, "prompt_tokens_details", None)
        )
        
        return cls(
            completion_tokens=response_obj.usage.completion_tokens,
            prompt_tokens=response_obj.usage.prompt_tokens,
            total_tokens=response_obj.usage.total_tokens,
            completion_tokens_details=completion_details,
            prompt_tokens_details=prompt_details,
            response_time=response_obj.usage.response_time
        )

class ModelConfig(BaseModel):
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    json_format: bool
    

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
    config: ModelConfig
    openai_api_key: Optional[str] = Field(
        None, description="API key if required by the openai provider."
    )
    images: Optional[List[str]] = Field(
        None, description="Images if using multimodal model"
    )
    anthropic_api_key: Optional[str] = Field(
        None, description="API key if required by the anthropic provider."
    )


class ModelInfo(BaseModel):
    provider: str = Field(..., description="Provider of the model.")
    model_name: str = Field(..., description="Name of the model.")


async def fetch_models_from_supabase() -> List[dict]:
    logging.info("Fetching models from Supabase")
    response = supabase.table("available_models").select("*").execute()
    logging.info(f"Fetched {len(response.data)} models")
    return response.data


@contextmanager
def temporary_env_var(key: str, value: str):
    """Context manager to temporarily set an environment variable."""
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value
        
def redact_words(model_name, text):
    for word in LIST_OF_REDACTED_WORDS:
        text = re.sub(rf"(?i)\b{re.escape(word)}\b", r"<redacted>\g<0></redacted>", text)
            
    return text

async def handle_completion(
    model_name: str, 
    message: str, 
    config: ModelConfig, 
    api_base: Optional[str] = None,
    images: Optional[List[str]] = None,
):
    try:
        start_time = time.time()
        
        messages = [
            {
                "role": "system",
                "content": config.system_prompt + (". Please generate the response in JSON" if config.json_format else ""),  
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    },
                ], 
            }
        ]
        if images:
            for image in images:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"{image}"},
                })

        if api_base:
            logging.info(f"Using API base: {api_base}")
            response_obj = completion(
                model=model_name,
                messages=messages,
                api_base=api_base + "/v1",
                timeout=180.00,
                metadata = {
                    "generation_name": model_name, # set langfuse generation name
                },
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                response_format= {"type": "json_object"} if config.json_format else None,
                api_key="None",
            )
        else:
            response_obj = completion(
                model=model_name,
                messages=messages,
                timeout=180.00,
                metadata = {
                    "generation_name": model_name, # set langfuse generation name
                },
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                response_format= {"type": "json_object"} if config.json_format == True else None
            )

        end_time = time.time()
        response_obj.usage.response_time = (end_time - start_time) * 1000
        
        # Check if response is empty and replace with default message
        content = response_obj.choices[0].message.content
        if not content or content.strip() == "":
            response_obj.choices[0].message.content = "Sorry, I couldn't answer this question :("
        else:
            response_obj.choices[0].message.content = redact_words(model_name, content)
            
        # Convert the usage object
        response_obj.usage = Usage.from_response(response_obj)
        
        return response_obj
    except OpenAIError as e:
        error_msg = str(e)
        logging.error(f"Error during completion: {error_msg}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Error during completion",
                "message": error_msg
            }
        )
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Unexpected error during completion: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Unexpected error during completion",
                "message": error_msg
            }
        )


@web_app.post(
    "/message",
    response_model=ModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Send message to a model",
    responses={
        200: {"description": "Successful response with the model's reply."},
        400: {"description": "Bad Request. Model or message not provided."},
        404: {"description": "Model is not supported."},
        500: {"description": "Internal Server Error."},
    },
)
async def messaging(request: MessageRequest):
    logging.info("Received /message request")
    model_name = request.model
    message = request.message
    openai_api_key = request.openai_api_key
    anthropic_api_key = request.anthropic_api_key
    config = request.config
    images = request.images
    
    logging.info(f"Requested model name: {model_name}")
    logging.info(f"Config: {config}")
    logging.info(f"Message: {message}")
    # Fetch models from Supabase
    models = await fetch_models_from_supabase()

    # OpenAI provider check
    openai_model = next((m for m in models if m["model_name"] == model_name and m["provider"] == "openai"), None)
    if openai_model and openai_api_key:
        logging.info(f"Using OpenAI provider with model_id: {openai_model['model_id']}")
        with temporary_env_var("OPENAI_API_KEY", openai_api_key):
            return await handle_completion(openai_model['model_id'], message, config=config, images=images)

    # Anthropic provider check
    anthropic_model = next((m for m in models if m["model_name"] == model_name and m["provider"] == "anthropic"), None)
    if anthropic_model and anthropic_api_key:
        logging.info(f"Using Anthropic provider with model_id: {anthropic_model['model_id']}")
        with temporary_env_var("ANTHROPIC_API_KEY", anthropic_api_key):
            return await handle_completion(anthropic_model['model_id'], message, config=config, images=images)

    # GitHub provider check
    github_model = next((m for m in models if m["model_name"] == model_name and m["provider"] == "github"), None)
    if github_model:
        logging.info(f"Using GitHub provider with model_id: {github_model['model_id']}")
        model_id = f"{github_model['model_id']}"
        return await handle_completion(model_id, message, config=config, images=images)

    # Hugging Face provider check
    huggingface_model = next((m for m in models if m["model_name"] == model_name and m["provider"] == "huggingface"), None)
    if huggingface_model:
        logging.info(f"Using Hugging Face provider with model_id: {huggingface_model['model_id']}")
        model_id = f"{huggingface_model['model_id']}"
        return await handle_completion(model_id, message, config=config, images=images)

    # Ollama provider check
    ollama_model = next((m for m in models if m["model_name"] == model_name and m["provider"] == "ollama"), None)
    if ollama_model:
        logging.info(f"Using Ollama provider with model_id: {ollama_model['model_id']}")
        model_id = f"openai/{ollama_model['model_id']}"
        api_url = os.environ['OLLAMA_API_URL']
        return await handle_completion(model_id, message, config=config, api_base=api_url, images=images)

    # Error handling
    model_info = next((m for m in models if m["model_name"] == model_name), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not supported")
    elif model_info["provider"] in ["openai", "anthropic"]:
        raise HTTPException(status_code=400, detail=f"API key required for {model_info['provider']} model")
    else:
        raise HTTPException(status_code=400, detail="Provider not supported")


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
