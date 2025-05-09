import os
import logging
import time
from typing import List, Optional, AsyncGenerator, Any, Union
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import contextmanager
from modal import Image, App, asgi_app, Secret
from const import LIST_OF_REDACTED_WORDS
import re
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
    "https://supa-rlhf.vercel.app",
    "https://develop.d3s6vhvxthx2m9.amplifyapp.com",
    "https://glhf.supa.so"
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
    ["litellm", "supabase", "pydantic==2.5.3", "fastapi==0.109.0", "openai", "langfuse", "huggingface-hub"]
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
        Secret.from_name("cloudflare-developers"),
    ],
)

with llm_compare_app.image.imports():
    import litellm
    from litellm import completion, acompletion
    from supabase import create_client, Client
    from openai import OpenAIError
    from huggingface_hub import repo_info
    from huggingface_hub.utils import RepositoryNotFoundError
    import re

    litellm.set_verbose = True
    litellm.drop_params = True
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


class Config(BaseModel):
    system_prompt: Optional[str] = Field(
        None, description="Custom system prompt for the model."
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature for the model."
    )
    top_p: Optional[float] = Field(1, description="Top-p sampling parameter.")
    max_tokens: Optional[int] = Field(
        1000, description="Maximum number of tokens in the response."
    )
    json_format: Optional[bool] = Field(
        False, description="Whether to format the response as JSON."
    )
    stream: Optional[bool] = Field(False, description="Whether to stream the response.")


class MessageRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use.")
    message: str = Field(..., description="Message text to send to the model.")
    images: List[str] = Field([], description="Array of base64-encoded images.")
    config: Optional[Config] = Field(
        None, description="Configuration parameters for the model request."
    )
    openai_api_key: Optional[str] = Field(
        None, description="API key if required by the openai provider."
    )
    anthropic_api_key: Optional[str] = Field(
        None, description="API key if required by the anthropic provider."
    )
    huggingface_token: Optional[str] = Field(
        None, description="API key if required by the huggingface provider."
    )
    huggingface_api_base: Optional[str] = Field(
        None, description="Custom and paid API base for the huggingface provider to use."
    )
    huggingface_repo_id: Optional[str] = Field(
        None, description="Repository ID for the huggingface provider. Often seen as [model_owner]/[model_name]"
    )


class Question(BaseModel):
    question: str = Field(..., description="The question text")
    tags: List[str] = Field(..., description="List of tags for the question")


class QuestionGenerationRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use.")
    input_question: Optional[Question] = Field(
        None, description="Single question with optional tags"
    )
    tag_limit: Optional[int] = Field(
        None, description="Maximum number of tags to use for generated questions"
    )
    openai_api_key: Optional[str] = Field(
        None, description="API key if required by the openai provider."
    )
    anthropic_api_key: Optional[str] = Field(
        None, description="API key if required by the anthropic provider."
    )
    huggingface_token: Optional[str] = Field(
        None, description="API key if required by the huggingface provider."
    )
    huggingface_api_base: Optional[str] = Field(
        None, description="Custom and paid API base for the huggingface provider to use."
    )
    huggingface_repo_id: Optional[str] = Field(
        None, description="Repository ID for the huggingface provider. Often seen as [model_owner]/[model_name]"
    )


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
            "examples": [
                {
                    "questions": [
                        {"question": "Example question text", "tags": ["example_tag"]}
                    ]
                }
            ]
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


def strip_thinking_tags(content: str) -> str:
    """Strip <think>...</think> tags and their contents from the response. WIP"""
    if isinstance(content, str):
        # Pattern to match <think>...</think> including newlines
        pattern = r"<think>[\s\S]*?</think>"
        # Remove all instances of the pattern
        cleaned_content = re.sub(pattern, "", content)
        return cleaned_content
    return content


async def handle_completion(
    model_name: str,
    message: str,
    config: Optional[Config] = None,
    api_base: Optional[str] = None,
    output_struct: Optional[BaseModel] = None,
    images: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    stream: Optional[bool] = False,
) -> Union[Any, AsyncGenerator[str, None]]:
    try:
        start_time = time.time()

        # Determine if we're using Cloudflare provider
        is_cloudflare = "cloudflare/" in model_name if model_name else False

        # Determine the appropriate system prompt
        system_content = """You are a helpful AI assistant."""

        if output_struct == QuestionResponse:
            # Use special system prompt for question generation with tags
            system_content = """You are a JSON-only API. Always respond with valid JSON matching the required schema. 
Never include explanatory text outside the JSON structure.
CRITICAL: Response must be a valid JSON object, not a string containing JSON.
CRITICAL REQUIREMENT: Each question must have EXACTLY the number of tags specified in the prompt.
For initial questions with no input, use EXACTLY ONE tag per question.
Never provide more tags than requested."""
        elif config and config.json_format:
            # Generic JSON system prompt without tag-specific instructions
            system_content = """You are a JSON-only API. Always respond with valid JSON matching the required format.
Never include explanatory text outside the JSON structure.
Your response must be a valid JSON object, not a string containing JSON."""

        # Override with custom system prompt if provided
        if config and config.system_prompt:
            system_content = config.system_prompt
            # Ensure the word "json" is present when json_format is true
            if config.json_format and "json" not in system_content.lower():
                system_content += "\nPlease format your response as JSON."

        # Prepare the user message content - handle multimodal input if images are provided
        user_content = message

        # If JSON format is requested but the word "json" is not in the messages,
        # append a request for JSON to the user message
        if (
            config
            and config.json_format
            and "json" not in message.lower()
            and "json" not in system_content.lower()
        ):
            if isinstance(user_content, str):
                user_content += "\nPlease format your response as JSON."

        if images and len(images) > 0:
            # Format multimodal content as an array of content objects
            if isinstance(user_content, str):
                user_content = [{"type": "text", "text": user_content}]

            # Add each image as an image_url object
            for image in images:
                user_content.append({"type": "image_url", "image_url": {"url": image}})

        completion_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "timeout": 300.00,
            "metadata": {
                "generation_name": model_name,
            },
            "stream": stream,
        }

        # Apply config parameters if provided
        if config:
            if config.temperature is not None:
                completion_kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                completion_kwargs["top_p"] = config.top_p
            if config.max_tokens is not None:
                completion_kwargs["max_tokens"] = config.max_tokens
            if config.stream is not None:
                completion_kwargs["stream"] = config.stream
                if config.stream:
                    completion_kwargs["stream_options"] = {"include_usage": True}

        # Set response format based on the scenario and provider
        if output_struct:
            json_schema = output_struct.model_json_schema()
            if is_cloudflare:
                # For Cloudflare, use simpler response format to avoid schema issues
                completion_kwargs["response_format"] = {"type": "json_object"}
            else:
                # Full JSON schema validation for other providers
                completion_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                    "strict": True,
                }
        elif config and config.json_format:
            # Simple JSON response format for generic JSON requests
            completion_kwargs["response_format"] = {"type": "json_object"}

        if api_base:
            # For Ollama's OpenAI-compatible endpoint, ensure we use the correct path and provide a dummy API key
            if "openai" in model_name.lower():
                # Ensure the base URL points to /v1 endpoint
                if not api_base.endswith("/v1"):
                    api_base = f"{api_base.rstrip('/')}/v1"
                completion_kwargs["api_base"] = api_base
                completion_kwargs["api_key"] = (
                    "ollama"  # Required but not validated by Ollama
                )
            else:
                completion_kwargs["api_base"] = api_base
                
        # Handle streaming response
        if stream or (config and config.stream):
            async def stream_generator():
                try:
                    content_buffer = ""
                    async for chunk in await acompletion(**completion_kwargs):
                        # Extract only the content from the chunk, filtering out metadata
                        content = None
                        
                        # Check for usage data in the chunk (both object and dict formats)
                        usage_data = None
                        if hasattr(chunk, "usage") and chunk.usage:
                            end_time = time.time()
                            chunk.usage.response_time = (end_time - start_time) * 1000
                            # Handle ModelResponse object with usage attribute
                            usage_data = chunk.usage.dict() if hasattr(chunk.usage, "dict") else chunk.usage
                        elif isinstance(chunk, dict) and "usage" in chunk:
                            end_time = time.time()
                            chunk["usage"].response_time = (end_time - start_time) * 1000
                            # Handle dictionary with usage key
                            usage_data = chunk["usage"]
                        
                        # If usage data is present, yield it as metadata
                        if usage_data:
                            metadata = f"<metadata>{json.dumps(usage_data)}</metadata>"
                            yield metadata
                            # Continue to next chunk if this was just a usage chunk
                            if (hasattr(chunk, "choices") and not chunk.choices) or \
                               (isinstance(chunk, dict) and "choices" in chunk and not chunk["choices"]):
                                continue
                        
                        # Convert ModelResponse to a serializable format
                        if hasattr(chunk, "choices") and chunk.choices:
                            # Standard OpenAI-like format
                            if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                                content = chunk.choices[0].delta.content
                            elif hasattr(chunk.choices[0], "message") and hasattr(chunk.choices[0].message, "content"):
                                content = chunk.choices[0].message.content
                        elif isinstance(chunk, dict):
                            # Dict format
                            if "choices" in chunk and chunk["choices"]:
                                choices = chunk["choices"]
                                if isinstance(choices, list) and choices:
                                    if "delta" in choices[0] and "content" in choices[0]["delta"] and choices[0]["delta"]["content"]:
                                        content = choices[0]["delta"]["content"]
                                    elif "message" in choices[0] and "content" in choices[0]["message"]:
                                        content = choices[0]["message"]["content"]
                            
                            # Check for usage at the end of stream
                            if "usage" in chunk:
                                metadata = f"<metadata>{json.dumps(chunk['usage'])}</metadata>"
                                yield metadata
                                continue
                        
                        # Only yield content if we found it, otherwise skip this chunk
                        if content:
                            content_buffer += content
                            yield content
                        
                        # For error cases, still return the error
                        if isinstance(chunk, dict) and "error" in chunk:
                            yield json.dumps({"error": chunk["error"]}) + "\n"
                except Exception as e:
                    logging.error(f"Error during streaming: {str(e)}")
                    yield json.dumps({"error": str(e)}) + "\n"
            return stream_generator()
        
        # Non-streaming response
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
            # Strip thinking tags specifically for question generation responses
            if output_struct == QuestionResponse and isinstance(content, str):
                content = strip_thinking_tags(content)

            # Apply redaction
            response_obj.choices[0].message.content = redact_words(model_name, content)

        # Convert the usage object
        response_obj.usage = Usage.from_response(response_obj)

        return response_obj
    except OpenAIError as e:
        error_msg = str(e)
        logging.error(f"Error during completion: {error_msg}")
        error_response = {
            "questions": [
                {"question": "Error occurred during completion", "tags": ["error"]}
            ]
        }
        raise HTTPException(status_code=500, detail=error_response)
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Unexpected error during completion: {error_msg}")
        error_response = {
            "questions": [{"question": "Unexpected error occurred", "tags": ["error"]}]
        }
        raise HTTPException(status_code=500, detail=error_response)


async def route_model_request(
    model_name: str,
    message: str,  # Expecting a string message
    config: Optional[Config] = None,
    images: Optional[List[str]] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    output_struct: Optional[BaseModel] = None,
    huggingface_token: Optional[str] = None,
    huggingface_repo_id: Optional[str] = None,
    huggingface_api_base: Optional[str] = None,

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
            return await handle_completion(
                openai_model["model_id"],
                message,
                config=config,
                images=images,
                output_struct=output_struct,
                stream=config.stream if config else False,
            )
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
            return await handle_completion(
                anthropic_model["model_id"],
                message,
                config=config,
                images=images,
                output_struct=output_struct,
                stream=config.stream if config else False,
            )

    # Hugging face provider check
    if huggingface_token and huggingface_repo_id:
        with temporary_env_var("HUGGINGFACE_API_KEY", huggingface_token):
            logging.info(f"Using Hugging Face provider with model_id: {huggingface_repo_id}")
            
            model_id = f"huggingface/{huggingface_repo_id}"
            
            return await handle_completion(
                model_id,
                message,
                config=config,
                images=images,
                output_struct=output_struct,
                stream=config.stream if config else False,
                api_base=huggingface_api_base,
            )
    
    cloudflare_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "cloudflare"
        ),
        None,
    )
    if cloudflare_model:
        logging.info(
            f"Using Cloudflare provider with model_id: {cloudflare_model['model_id']}"
        )
        model_id = f"cloudflare/{cloudflare_model['model_id']}"
        return await handle_completion(
            model_id,
            message,
            config=config,
            images=images,
            output_struct=output_struct,
            stream=config.stream if config else False,
        )

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
        return await handle_completion(
            github_model["model_id"],
            message,
            config=config,
            images=images,
            output_struct=output_struct,
            stream=config.stream if config else False,
        )

    # Hugging Face provider check for your own huggingface repo (not client provided credentials one)
    huggingface_model = next(
        (
            m
            for m in models
            if m["model_name"] == model_name and m["provider"] == "huggingface"
        ),
        None,
    )
    if huggingface_model and not huggingface_token:
        logging.info(
            f"Using Hugging Face provider with model_id: {huggingface_model['model_id']}"
        )
        return await handle_completion(
            huggingface_model["model_id"],
            message,
            config=config,
            images=images,
            output_struct=output_struct,
            stream=config.stream if config else False,
        )

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

        return await handle_completion(
            model_id,
            message,
            config=config,
            api_base=api_url,
            images=images,
            output_struct=output_struct,
            stream=config.stream if config else False,
        )

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


def get_required_tag_count(
    question: Optional[Question], tag_limit: Optional[int] = None
) -> int:
    """Determine the number of tags required based on input question."""
    # Default tag limit if none provided
    if tag_limit is None:
        tag_limit = 5

    if not question:
        return 1
    if not question.tags:
        return 1
    # Limit to n tags total (input tags + 1 new tag)
    if len(question.tags) >= tag_limit - 1:
        return tag_limit
    return len(question.tags) + 1


@web_app.post("/question_generation")
async def question_generation(request: QuestionGenerationRequest):
    required_tag_count = get_required_tag_count(
        request.input_question, request.tag_limit
    )

    # Modified system prompt to prefer concise questions
    system_prompt = f"""You are a precise AI assistant that creates clear, focused questions.
When generating questions with tags:
1. ALWAYS use EXACTLY {required_tag_count} tag(s) per question - no more, no fewer
2. Create concise questions that are only as detailed as necessary
3. Only use paragraph-length when complexity truly requires it
4. Format your response as a valid JSON object with the structure requested
5. Each question should be direct and to the point
6. Tags can be simple words OR short phrases (up to 5 words maximum)"""

    tag_requirements = """
Tag Requirements:
- EVERY question MUST have EXACTLY the specified number of tags
- Tags should be clear, relevant, and concise
- Tags can be single words OR short phrases (maximum 5 words)
- First tag should represent the primary capability/category
- Additional tags should add specificity

Question Quality Requirements:
- Questions should be concise yet detailed enough to be clear
- Only use paragraph-length when the complexity truly requires it
- Include necessary context, but be as brief as possible
- Provide clear instructions on what's expected
- Focus on topics where AI can demonstrate reasoning, creativity, or knowledge
- Avoid unnecessary verbosity or filler text
"""

    if not request.input_question:
        message = """Generate 4 diverse questions for evaluating language models.

Required JSON Schema:
{
    "questions": [
        {
            "question": "Concise question text that's only as detailed as necessary",
            "tags": ["capability"] // EXACTLY ONE TAG - NO EXCEPTIONS
        },
        ... // 3 more questions, each with EXACTLY ONE TAG
    ]
}

CRITICAL REQUIREMENTS:
1. Each question MUST have EXACTLY ONE TAG - no exceptions
2. Questions should be concise yet detailed enough to be clear
3. Only use paragraph-length for complex topics that require it
4. Keep questions focused and to the point
5. Create questions that test different cognitive abilities:
   - One testing analytical reasoning
   - One testing creative thinking
   - One testing knowledge application
   - One testing logical problem solving
6. NO questions requiring very recent information or real-time data

DOUBLE CHECK: I need EXACTLY 4 questions, each with EXACTLY 1 tag.
"""

    elif not request.input_question.tags:
        message = f"""Analyze this question and generate 4 related questions.

Input Question: "{request.input_question.question}"

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "{request.input_question.question}",
            "tags": ["primaryCapability"]
        }},
        ... // 3 more related questions, each with EXACTLY ONE TAG
    ]
}}

CRITICAL REQUIREMENTS:
1. First question must be the input question with one appropriate tag
2. Generate 3 additional related questions that explore similar themes or concepts
3. Each question MUST have EXACTLY ONE TAG - no exceptions
4. Keep questions concise yet clear and focused
5. Only use paragraph-length if absolutely necessary for clarity
6. Questions should be direct and to the point

{tag_requirements}"""
    else:
        # Get original tags, limiting to first 4 if more are provided
        original_tags = request.input_question.tags[:4]
        tag_list = ", ".join([f'"{tag}"' for tag in original_tags])
        new_tag_count = min(len(original_tags) + 1, 5)

        message = f"""Analyze this tagged question and generate 4 related questions.

Input Question: {{
    "question": "{request.input_question.question}",
    "tags": [{tag_list}]
}}

Required JSON Schema:
{{
    "questions": [
        {{
            "question": "Concise question that provides necessary context and clear instructions",
            "tags": [{tag_list}, "additionalTag"] // EXACTLY {new_tag_count} tags
        }},
        ... // 3 more questions, each with EXACTLY {new_tag_count} tags
    ]
}}

CRITICAL REQUIREMENTS:
1. EVERY question MUST have EXACTLY {new_tag_count} tags total
2. Keep ALL original tags: [{tag_list}] in the EXACT SAME order
3. Add EXACTLY ONE NEW relevant tag at the end (word or short phrase up to 5 words)
4. Make questions concise yet clear and detailed:
   - Only as long as necessary to provide context and instructions
   - Use paragraph-length ONLY when complexity truly requires it
   - Prefer shorter, more focused questions when possible
5. Questions must be related to the input theme but explore different aspects

DOUBLE-CHECK: Count the tags for each question - there should be EXACTLY {new_tag_count} tags per question."""

    message_text = f"{system_prompt}\n\n{message}"

    # Create a config with stream set to False for question generation
    config = Config(stream=False, json_format=True)
    
    return await route_model_request(
        model_name=request.model,
        message=message_text,
        config=config,
        openai_api_key=request.openai_api_key,
        anthropic_api_key=request.anthropic_api_key,
        output_struct=QuestionResponse,
    )


@web_app.post(
    "/message",
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

    if request.config and request.config.stream:
        stream_generator = await route_model_request(
            model_name=request.model,
            message=request.message,
            config=request.config,
            images=request.images,
            openai_api_key=request.openai_api_key,
            anthropic_api_key=request.anthropic_api_key,
            huggingface_token=request.huggingface_token,
            huggingface_repo_id=request.huggingface_repo_id,
            huggingface_api_base=request.huggingface_api_base,
        )
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
            }
        )
    else:
        return await route_model_request(
            model_name=request.model,
            message=request.message,
            config=request.config,
            images=request.images,
            openai_api_key=request.openai_api_key,
            anthropic_api_key=request.anthropic_api_key,
            huggingface_token=request.huggingface_token,
            huggingface_repo_id=request.huggingface_repo_id,
            huggingface_api_base=request.huggingface_api_base,
        )


@web_app.get(
    "/hf_model_validation",
    response_model=bool,
    status_code=status.HTTP_200_OK,
    summary="Validate if huggingface model entered exists and can be used",
    responses={
        200: {"description": "Model validation successful."},
        500: {"description": "Internal Server Error."},
    },
)
async def hf_model_validation(repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None) -> bool:
    logging.info(f"Requested model name: {repo_id}")
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False

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


@llm_compare_app.function(scaledown_window=1200)
@asgi_app()
def fastapi_app():
    logging.info("Starting FastAPI app")
    return web_app
