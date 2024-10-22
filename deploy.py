from ollama_service import ollama_app
from modal import App

app = App("llm-comparison-api")
app.include(ollama_app)