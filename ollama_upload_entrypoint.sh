#!/bin/bash

# Start Ollama service in background
ollama serve &

# Wait for Ollama service to be ready
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null; then
        echo "Ollama service is ready"
        break
    fi
    echo "Waiting for Ollama service... ($i/30)"
    sleep 2
done

# Execute the passed command
exec "$@"