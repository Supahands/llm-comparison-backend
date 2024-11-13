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

## HuggingFace to GGUF Converter

The project includes a HuggingFace to GGUF converter (`hugging_face_to_guff.py`) that enables converting HuggingFace models to GGUF format using Modal's infrastructure.

### Environment Variables

Required environment variables:
```sh
HUGGING_FACE_HUB_TOKEN="your_hf_token"  # HuggingFace API token with read access
```
Add these to your `.env` file or Modal secrets.

### Setup
1. Create a Modal secret for HuggingFace, either via the UI or the cli:
```sh
modal secret create my-huggingface-secret HUGGING_FACE_HUB_TOKEN="your_token"
```
2. Run the modal file using:
```sh
modal run --detach hugging_face_to_guff.py --modelowner tencent --modelname Tencent-Hunyuan-Large --quanttype q8_0  --username Supa-AI
```
- The `--detach` command is used to allow this program to run even if your terminal disconnects from the modal servers
- `modelowner` is the repo owner that you are trying to get the model from
- `modelname` is the exact name of the model from that model owner you want to convert
- `quanttype` is the size of quantization, default is `q8_0` which is the largest this supports 
- `username` is used to determine which account it should upload to and create a repo for

## Technical Details
### Storage
- Models are stored on Modal volumes (model-storage) which will be created for you on running this modal file
- Large models (>10GB) may take significant time to download
- Volumes persist between runs, and it should be able to detect which files it has downloaded previously to then either continue downloading or directly begin to prcess

### Key Components of this program
### Fast downloads
- Uses custom HuggingFace downloader from [booday](https://github.com/bodaay/HuggingFaceModelDownloader) 
- Supports parallel downloads (8 connections)
- Includes progress tracking and ETA estimation

### Conversion Process 
- Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion
- Two-step process:
    1. Convert to FP16 format
    2. Quantize to desired format (Q4_K_M, Q5_K_M etc)
- Supports importance matrix for optimized quantization
- Can split large models into manageable shards
