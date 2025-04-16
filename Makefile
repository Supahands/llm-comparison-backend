# Makefile for llm-comparison-backend (Modal app)

.PHONY: setup deploy serve

# Setup the environment (installs modal)
setup:
	pip install --upgrade pip
	pip install modal
	modal setup

# Deploy the Modal app (deploy.py is the main entrypoint)
deploy:
	modal deploy deploy.py

# Serve the Modal app locally (for dev/testing)
serve:
	modal serve deploy.py
