import os
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import contextmanager
from modal import Image, App, asgi_app, Secret
from const import LIST_OF_REDACTED_WORDS
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

os.environ["LITELLM_LOG"] = "DEBUG"

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
    ["litellm", "supabase", "pydantic==2.5.3", "fastapi==0.109.0", "openai", "langfuse"]
)
llm_compare_app = App(
    name="llm-compare-api",
    image=image,
    secrets=[
        Secret.from_name("SUPABASE_SECRETS"),
        Secret.from_name("OLLAMA_API"),
        Secret.from_name("llm_comparison_github"),
        Secret.from_name("my-huggingface-secret"),
        Secret.from_name("Langfuse-Secret"),
    ],
)

with llm_compare_app.image.imports():
    import litellm
    from litellm import completion
    from supabase import create_client, Client
    from openai import OpenAIError
    import re

    litellm.set_verbose = True
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]  # logs errors to langfuse

    # Initialize Supabase client
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)

    # Enable JSON schema validation
    litellm.enable_json_schema_validation = True


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
    completion_tokens_details: Optional[dict] = Field(
        default=None, description="Details about completion tokens"
    )
    prompt_tokens_details: Optional[dict] = Field(
        default=None, description="Details about prompt tokens"
    )
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
            response_time=response_obj.usage.response_time,
        )


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
    openai_api_key: Optional[str] = Field(
        None, description="API key if required by the openai provider."
    )
    anthropic_api_key: Optional[str] = Field(
        None, description="API key if required by the anthropic provider."
    )


class Question(BaseModel):
    question: str = Field(..., description="The question text")
    tags: List[str] = Field(..., description="List of tags for the question")


class QuestionGenerationRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use.")
    input_question: Optional[Question] = Field(None, description="Single question with optional tags")
    openai_api_key: Optional[str] = Field(None, description="API key if required by the openai provider.")
    anthropic_api_key: Optional[str] = Field(None, description="API key if required by the anthropic provider.")


class ModelInfo(BaseModel):
    provider: str = Field(..., description="Provider of the model.")
    model_name: str = Field(..., description="Name of the model.")


class GeneratedQuestion(BaseModel):
    question: str
    tags: List[str]


class QuestionResponse(BaseModel):
    questions: List[GeneratedQuestion]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "questions": [
                    {
                        "question": "Example question text",
                        "tags": ["example_tag"]
                    }
                ]
            }]
        }
    }


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
        text = re.sub(
            rf"(?i)\b{re.escape(word)}\b", r"<redacted>\g<0></redacted>", text
        )

    return text


async def handle_completion(
    model_name: str,
    message: str,
    api_base: Optional[str] = None,
    output_struct: Optional[BaseModel] = None,
):
    try:
        start_time = time.time()
        completion_kwargs = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant.""" if not output_struct else
                    """You are a JSON-only API. Always respond with valid JSON matching the required schema. 
Never include explanatory text outside the JSON structure.
CRITICAL: Response must be a valid JSON object, not a string containing JSON.
CRITICAL REQUIREMENT: Each question must have EXACTLY the number of tags specified in the prompt.
For initial questions with no input, use EXACTLY ONE tag per question.
Never provide more tags than requested."""
                },
                {"role": "user", "content": message}
            ],
            "timeout": 180.00,
            "metadata": {
                "generation_name": model_name,
            },
        }

        # Only add JSON schema validation for question generation endpoint
        if output_struct:
            json_schema = output_struct.model_json_schema()
            completion_kwargs["response_format"] = {
                "type": "json_schema",
                "schema": json_schema,
                "strict": True
            }

        if api_base:
            # For Ollama's OpenAI-compatible endpoint, ensure we use the correct path and provide a dummy API key
            if "openai" in model_name.lower():
                # Ensure the base URL points to /v1 endpoint
                if not api_base.endswith('/v1'):
                    api_base = f"{api_base.rstrip('/')}/v1"
                completion_kwargs["api_base"] = api_base
                completion_kwargs["api_key"] = "ollama"  # Required but not validated by Ollama
            else:
                completion_kwargs["api_base"] = api_base

        if output_struct:
            # Convert Pydantic model to JSON schema
            json_schema = output_struct.model_json_schema()
            completion_kwargs["response_format"] = {
                "type": "json_schema",
                "schema": json_schema,
                "strict": True
            }

        response_obj = completion(**completion_kwargs)

        end_time = time.time()
        response_obj.usage.response_time = (end_time - start_time) * 1000

        # Check if response is empty and replace with default message
        content = response_obj.choices[0].message.content
        if not content or content.strip() == "":
            response_obj.choices[
                0
            ].message.content = "Sorry, I couldn't answer this question :("
        else:   
            response_obj.choices[0].message.content = redact_words(model_name, content)

        # Convert the usage object
        response_obj.usage = Usage.from_response(response_obj)

        return response_obj
    except OpenAIError as e:
        error_msg = str(e)
        logging.error(f"Error during completion: {error_msg}")
        error_response = {
            "questions": [
                {
                    "question": "Error occurred during completion",
                    "tags": ["error"]
                }
            ]
        }
        raise HTTPException(
            status_code=500,
            detail=error_response
        )
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Unexpected error during completion: {error_msg}")
        error_response = {
            "questions": [
                {
                    "question": "Unexpected error occurred",
                    "tags": ["error"]
                }
            ]
        }
        raise HTTPException(
            status_code=500,
            detail=error_response
        )


async def route_model_request(
    model_name: str,
    message: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    output_struct: Optional[str] = None,
):
    """Route the request to the appropriate model provider and handle API keys."""
    models = await fetch_models_from_supabase()

    # OpenAI provider check
    openai_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "openai"
        ),
        None,
    )
    if openai_model and openai_api_key:
        logging.info(f"Using OpenAI provider with model_id: {openai_model['model_id']}")
        with temporary_env_var("OPENAI_API_KEY", openai_api_key):
            return await handle_completion(openai_model["model_id"], message, output_struct=output_struct)

    # Anthropic provider check
    anthropic_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "anthropic"
        ),
        None,
    )
    if anthropic_model and anthropic_api_key:
        logging.info(
            f"Using Anthropic provider with model_id: {anthropic_model['model_id']}"
        )
        with temporary_env_var("ANTHROPIC_API_KEY", anthropic_api_key):
            return await handle_completion(anthropic_model["model_id"], message, output_struct=output_struct)

    # GitHub provider check
    github_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "github"
        ),
        None,
    )
    if github_model:
        logging.info(f"Using GitHub provider with model_id: {github_model['model_id']}")
        return await handle_completion(github_model["model_id"], message, output_struct=output_struct)

    # Hugging Face provider check
    huggingface_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "huggingface"
        ),
        None,
    )
    if huggingface_model:
        logging.info(
            f"Using Hugging Face provider with model_id: {huggingface_model['model_id']}"
        )
        return await handle_completion(huggingface_model["model_id"], message, output_struct=output_struct)

    # Ollama provider check
    ollama_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "ollama"
        ),
        None,
    )
    if ollama_model:
        logging.info(f"Using Ollama provider with model_id: {ollama_model['model_id']}")
        model_id = f"openai/{ollama_model['model_id']}"
        api_url = os.environ["OLLAMA_API_URL"]
        # api_url = "https://supa-dev--llm-comparison-api-ollama-api-dev.modal.run"
        return await handle_completion(model_id, message, api_base=api_url, output_struct=output_struct)

    # Error handling
    model_info = next((m for m in models if m["model_name"] == model_name), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not supported")
    elif model_info["provider"] in ["openai", "anthropic"]:
        raise HTTPException(
            status_code=400,
            detail=f"API key required for {model_info['provider']} model",
        )
    else:
        raise HTTPException(status_code=400, detail="Provider not supported")


def get_required_tag_count(question: Optional[Question]) -> int:
    """Determine the number of tags required based on input question."""
    if not question:
        return 1
    if not question.tags:
        return 1
    # Limit to 5 tags total (input tags + 1 new tag)
    if len(question.tags) >= 4:
        return 5
    return len(question.tags) + 1

@web_app.post("/question_generation")
async def question_generation(request: QuestionGenerationRequest):
    required_tag_count = get_required_tag_count(request.input_question)

    tag_requirements = """
Tag Requirements:
- Tags MUST be SINGLE WORDS ONLY
- NO hyphens, spaces, or special characters
- NO compound words or phrases
- Each tag should be a simple category word
- First tag is broadest category
- Additional tags narrow down the category
- Valid examples:
  * 'reasoning'
  * ['logic', 'mathematics']
  * ['science', 'physics', 'mechanics']
- Invalid examples:
  * 'risk-assessment'
  * 'logical_reasoning'
  * 'problem solving'"""

    if not request.input_question:
        message = f"""Generate 4 diverse questions for evaluating language models.

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "The question text goes here",
            "tags": ["capability"]  // EXACTLY ONE TAG PER QUESTION - NO EXCEPTIONS
        }}
    ]
}}

STRICT Requirements:
1. Each question MUST have EXACTLY ONE TAG - no more, no less
2. Use these categories only: reasoning, creativity, knowledge, logic, analysis
3. Clear questions
4. Tags must be single words
5. Challenging but answerable
6. No harmful topics

{tag_requirements}

CRITICAL: Any response with more than one tag per question will be rejected."""

    elif not request.input_question.tags:
        message = f"""Analyze this question and generate 4 related questions.

Input Question: "{request.input_question.question}"

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "{request.input_question.question}",
            "tags": ["primaryCapability"]
        }}
    ]
}}

Requirements:
- First question in response must be the input question with appropriate tag
- Generate 3 additional related questions
- Each question MUST have exactly 1 tag identifying its primary capability
- Related questions should test similar capabilities
- No harmful topics

{tag_requirements}"""
    else:
        # Get original tags, limiting to first 4 if more are provided
        original_tags = request.input_question.tags[:4]
        tag_list = ", ".join([f'"{tag}"' for tag in original_tags])
        new_tag_count = min(len(original_tags) + 1, 5)
        
        message = f"""Analyze this tagged question and generate 4 related questions with additional detail.

Input Question: {{
    "question": "{request.input_question.question}",
    "tags": [{tag_list}]
}}

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "The question text goes here",
            "tags": [{tag_list}, "additionalTag"]  // Original tags plus one new tag
        }}
    ]
}}

Requirements:
- Generate 4 new questions related to the input question
- Each question MUST have exactly {new_tag_count} tags total
- PRESERVE ALL original tags: [{tag_list}] in the exact same order
- Add EXACTLY ONE NEW complementary tag at the end of the tag list
- The new tag should describe a specific capability or aspect being tested
- Maximum of 5 tags total allowed
- If input has 4 tags already, new tag should be a refinement of the last tag
- Build upon the theme of the input question
- Explore different aspects while maintaining thematic connection
- No harmful topics

{tag_requirements}

CRITICAL:
- Keep original tags [{tag_list}] in their exact order
- Add exactly ONE new tag at the end
- Total tag count must be {new_tag_count} (maximum 5)
- Do not modify or remove any original tags
- If input has 4+ tags, new tag should refine/specialize the last tag"""

    return await route_model_request(
        request.model,
        message,
        request.openai_api_key,
        request.anthropic_api_key,
        output_struct=QuestionResponse
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
    logging.info(f"Requested model name: {request.model}")
    logging.info(f"Message: {request.message}")

    return await route_model_request(
        request.model,
        request.message,
        request.openai_api_key,
        request.anthropic_api_key,
    )


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
