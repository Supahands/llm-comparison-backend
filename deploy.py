from ollama_service import ollama_app
from ai_router import llm_compare_app
from modal import App

app = App("llm-comparison-api")
app.include(ollama_app)
app.include(llm_compare_app)