# llm-comparison-backend
This is an opensource project allowing you to compare two LLM's head to head with a given prompt, this section will be regarding the backend of this project, allowing for llm api's to be incorporated and used in the front-end

# Project Deployment

This project uses [Modal](https://modal.com/) to deploy Python applications as containerized services.

## Automatic Deployment

The project includes CI/CD automation via GitHub Actions. When code is pushed to the `main` branch, the workflow in [.github/workflows/deploy.yml](.github/workflows/deploy.yml) automatically:

1. Sets up Python 3.10
2. Installs dependencies via Poetry
3. Deploys the application using Modal

## Manual Deployment

### Prerequisites
- Python 3.10+
- Poetry for dependency management
- Modal CLI and account

### Setup
1. Install dependencies:
```sh 
poetry install
```
2. Configure Modal credentials:
- Copy .env.example to .env
- Add your Modal tokens from the Modal dashboard
### Deployment Commands
- **Production Deploy**:
```sh
modal deploy --env dev deploy
```

This command uses `deploy.py` as an entry point to bundle all application components into a single Modal deployment. The `deploy.py` file orchestrates the initialization and configuration of all microservices.

- **Local Testing:**
```sh
modal serve --env dev deploy
```

This creates a temporary deployment for testing purposes. The service remains active only while the command is running, making it ideal for development and testing.

The `deploy.py` file acts as the main orchestrator, combining all application components (like AI routing and Ollama services) into a unified Modal deployment. It handles the configuration and initialization of each service component within the Modal infrastructure. 