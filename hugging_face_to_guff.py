# This code is a modified version of https://huggingface.co/spaces/ggml-org/gguf-my-repo/
import os
import logging
from modal import Image, Volume, Secret, App, method
from typing import Union, List, Optional
import tempfile
import subprocess

# Supported quantization types
SUPPORTED_QUANT_TYPES = [
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q3_K_S",
    "q3_K_M",
    "q3_K_L",
    "q4_K_S",
    "q4_K_M",
    "q5_K_S",
    "q5_K_M",
    "q6_K",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = App("hf-to-gguf-converter")

# Create Modal volume
volume = Volume.from_name("model-storage", create_if_missing=True)
ollama_config_volume = Volume.from_name("ollama", create_if_missing=True)

# Base image with required dependencies and Ollama installation
image = (
    Image.from_registry("ollama/ollama:latest", add_python="3.11")
    .apt_install("git", "curl", "build-essential", "cmake")
    .pip_install(
        "huggingface_hub",
        "torch",
        "transformers",
        "sentencepiece",
        "accelerate",
        "tqdm",
    )
    .run_commands(
        # Download and install HF downloader
        "curl -sSL https://g.bodaay.io/hfd > /tmp/hfd.sh",
        "bash /tmp/hfd.sh -i",
        # Clone and build llama.cpp
        "cd /root && git clone https://github.com/ggerganov/llama.cpp.git",
        "cd /root/llama.cpp && make",
        "cd /root/llama.cpp && pip install -r requirements.txt",
    )
    .copy_local_file("./ollama_upload_entrypoint.sh", "/opt/entrypoint.sh")
    .dockerfile_commands(
        [
            "RUN chmod a+x /opt/entrypoint.sh",
            'ENTRYPOINT ["/opt/entrypoint.sh"]',
        ]
    )
)


@app.cls(
    image=image,
    secrets=[Secret.from_name("my-huggingface-secret")],
    volumes={"/root/models": volume, "/root/.ollama": ollama_config_volume},
    timeout=86400,
    gpu="A10G",
)
class ModelConverter:
    @method()
    def download_model(
        self,
        model_id: str,
        modelname: str,
        username: str,
        branch: str = "",
        filter_path: str = "",
        quanttypes: Union[str, List[str]] = "q8_0",
        private: bool = False,
        ollama_upload: bool = False,
        hf_upload: bool = False,
        clean_run: bool = False,  
    ):
        logger.info(f"Downloading model {model_id}...")
        import subprocess

        try:
            # Clean model storage if clean_run is True
            if clean_run:
                logger.info("Clean run requested - cleaning model storage volume...")
                models_dir = "/root/models"
                if os.path.exists(models_dir):
                    import shutil
                    # Remove all contents of models directory
                    for item in os.listdir(models_dir):
                        item_path = os.path.join(models_dir, item)
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    logger.info("Cleaned model storage directory")
                    # Commit volume after cleanup
                    volume.commit()

            local_dir = f"/root/models/{model_id.split('/')[-1]}-hf"
            os.makedirs(local_dir, exist_ok=True)

            cmd = [
                "hfdownloader",
                "-m",
                model_id,
                "-s",
                local_dir,
                "-k",
                "-c",
                "8",
            ]

            # Add branch if specified
            if branch:
                cmd.extend(["-b", branch])

            # Add filter if specified
            if filter_path:
                cmd.extend(["-f", filter_path])

            logging.info(f"token: {os.environ.get('HUGGINGFACE_API_KEY', '')}")
            # Add token at the end
            cmd.extend(["-t", os.environ.get("HUGGINGFACE_API_KEY", "")])

            # Run download command
            result = os.system(" ".join(cmd))
            if result != 0:
                raise Exception(f"Download failed with exit code {result}")

            logger.info(f"Model downloaded to {local_dir}")

            # Commit volume after download
            volume.commit()
            logger.info("Volume committed after download")

            # Call convert_to_gguf with .remote()
            return self.convert_to_gguf.remote(
                local_dir,
                modelname,
                quanttypes,
                model_id,
                username,
                private,
                branch,
                filter_path,
                ollama_upload,
                hf_upload,
                clean_run,  # Pass through
            )

        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

    @method()
    def convert_to_gguf(
        self,
        input_dir: str,
        modelname: str,
        quanttypes: Union[str, List[str]],
        source_model_id: str,
        username: str,
        private: bool,
        branch: str = "",
        filter_path: str = "",
        ollama_upload: bool = False,
        hf_upload: bool = False,  
        clean_run: bool = False,  
    ):
        """Convert model to GGUF format with multiple quantization types and push to Ollama"""
        logger.info(f"Converting model with quantization types: {quanttypes}")

        if isinstance(quanttypes, str):
            quanttypes = [quanttypes]

        if not quanttypes:
            quanttypes = SUPPORTED_QUANT_TYPES

        try:
            # Find model directory
            model_subdir = None
            for root, dirs, files in os.walk(input_dir):
                if "config.json" in files:
                    model_subdir = root
                    break

            if not model_subdir:
                raise Exception("Could not find config.json in model directory")

            logger.info(f"Found model files in: {model_subdir}")

            # First convert to f16 (only need to do this once)
            fp16_path = f"/root/models/{modelname}.f16.gguf"
            if not os.path.exists(fp16_path):
                logger.info("Converting to fp16 first...")
                cmd = f"python /root/llama.cpp/convert_hf_to_gguf.py {model_subdir} --outfile {fp16_path} --outtype f16"
                result = os.system(cmd)
                if result != 0:
                    raise Exception(f"FP16 conversion failed with code {result}")

            # Process each quantization type
            model_files = []
            for quanttype in quanttypes:
                try:
                    output_file = f"/root/models/{modelname}.{quanttype.lower()}.gguf"

                    # Skip if already exists
                    if os.path.exists(output_file):
                        logger.info(f"Quantized model already exists at {output_file}")
                    else:
                        # Quantize
                        logger.info(f"Quantizing to {quanttype}...")
                        cmd = f"/root/llama.cpp/llama-quantize {fp16_path} {output_file} {quanttype}"
                        result = os.system(cmd)
                        if result != 0:
                            logger.error(f"Quantization failed for {quanttype}")
                            continue

                        logger.info(f"Model converted and quantized to {output_file}")
                        volume.commit()

                    model_files.append((output_file, quanttype))

                except Exception as e:
                    logger.error(f"Error processing {quanttype}: {str(e)}")
                    continue

            # Upload all versions to Ollama if ollama_upload is True
            if ollama_upload:
                self.push_to_ollama.remote(
                    model_files, modelname, source_model_id, username, clean_run
                )

            # You can also upload to Hugging Face if needed
            if hf_upload:
                return self.upload_to_hf.remote(
                    model_files,
                    f"{username}/{modelname}-gguf",
                    source_model_id,
                    private,
                )

        except Exception as e:
            logger.error(f"Error in conversion/quantization: {str(e)}")
            raise
    
    @method()
    def upload_to_hf(self, model_files: List[tuple], repo_id: str, source_model_id: str, private: bool = False):
        logger.info("Reloading volume before upload...")
        volume.reload()
        
        logger.info(f"Uploading GGUF models to HuggingFace repo {repo_id}...")
        from huggingface_hub import HfApi, ModelCard
        from textwrap import dedent
        
        try:
            api = HfApi()
            api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
            
            try:
                card = ModelCard.load(source_model_id)
            except Exception:
                card = ModelCard("")
                
            if card.data.tags is None:
                card.data.tags = []
            card.data.tags.extend(["llama-cpp", "gguf", "quantized", "llm", "llama", "ollama"])
            card.data.base_model = source_model_id
            
            # Generate model card with all versions
            versions_text = "\n".join([
                f"- `{os.path.basename(file)}` ({quant_type})" 
                for file, quant_type in model_files
            ])
            
            card.text = dedent(f"""
# {repo_id}
This model was converted to GGUF format from [`{source_model_id}`](https://huggingface.co/{source_model_id}) using llama.cpp.
Refer to the [original model card](https://huggingface.co/{source_model_id}) for more details on the model.

## Available Versions
{versions_text}

## Use with llama.cpp
Replace `FILENAME` with one of the above filenames.

### CLI:
```bash
llama-cli --hf-repo {repo_id} --hf-file FILENAME -p "Your prompt here"
```

### Server:
```bash
llama-server --hf-repo {repo_id} --hf-file FILENAME -c 2048
```

## Model Details
- **Original Model:** [{source_model_id}](https://huggingface.co/{source_model_id})
- **Format:** GGUF
            """)
            
            # Save and upload README
            readme_path = "/tmp/README.md"
            card.save(readme_path)
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id
            )
            
            # Upload all model files
            for file_path, _ in model_files:
                filename = os.path.basename(file_path)
                logger.info(f"Uploading quantized model: {filename}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id
                )
            
            # Upload imatrix.dat if it exists
            imatrix_path = "/root/llama.cpp/imatrix.dat"
            if os.path.isfile(imatrix_path):
                logger.info("Uploading imatrix.dat")
                api.upload_file(
                    path_or_fileobj=imatrix_path,
                    path_in_repo="imatrix.dat",
                    repo_id=repo_id
                )
                
            logger.info("Upload completed successfully")
            
        except Exception as e:
            logger.error(f"Error uploading to HuggingFace: {str(e)}")
            raise

    @method()
    def push_to_ollama(
        self,
        model_files: List[tuple],
        modelname: str,
        source_model_id: str,
        username: str,
        clean_run: bool = False,  
    ):
        """Push converted models to Ollama using tags for different quantizations"""
        logger.info("Pushing models to Ollama...")

        try:
            if clean_run:
                logger.info("Clean run requested - removing existing Ollama models...")
                ollama_models_dir = "/root/.ollama/models"
                if os.path.exists(ollama_models_dir):
                    import shutil
                    shutil.rmtree(ollama_models_dir)
                    os.makedirs(ollama_models_dir)
                    logger.info("Cleaned Ollama models directory")

            # Convert to lowercase for Ollama compatibility
            base_name = f"{username}/{modelname.lower()}"
            logger.info(f"Using lowercase model name: {base_name}")

            for model_path, quant_type in model_files:
                tag_name = quant_type.lower()

                # Verify file exists and is accessible
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    continue

                # Get absolute path
                model_path = os.path.abspath(model_path)
                logger.info(f"Using absolute model path: {model_path}")

                # Create Modelfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
                    modelfile_content = f"""FROM {model_path}
LICENSE Apache 2.0
TEMPLATE "{{{{.Input}}}}"
SYSTEM "You are a helpful AI assistant. Original model: {source_model_id}"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "User:"
PARAMETER stop "Assistant:"
"""
                    f.write(modelfile_content)
                    f.flush()
                    modelfile_path = f.name
                    logger.info(f"Created Modelfile at {modelfile_path}")

                try:
                    # Wait for Ollama service
                    import time

                    max_retries = 30
                    for i in range(max_retries):
                        try:
                            subprocess.run(
                                ["ollama", "list"], check=True, capture_output=True
                            )
                            break
                        except Exception:
                            if i == max_retries - 1:
                                raise Exception("Ollama service not responding")
                            logger.info(
                                f"Waiting for Ollama service... ({i+1}/{max_retries})"
                            )
                            time.sleep(1)

                    # Create model
                    logger.info(f"Creating Ollama model {base_name}:{tag_name}")
                    subprocess.run(
                        [
                            "ollama",
                            "create",
                            f"{base_name}:{tag_name}",
                            "-f",
                            modelfile_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    # Verify model was created
                    result = subprocess.run(
                        ["ollama", "list"], check=True, capture_output=True, text=True
                    )
                    if f"{base_name}:{tag_name}" not in result.stdout:
                        raise Exception(
                            f"Model {base_name}:{tag_name} not found after creation"
                        )

                    # Push model
                    logger.info(f"Pushing model {base_name}:{tag_name}")
                    push_result = subprocess.run(
                        ["ollama", "push", f"{base_name}:{tag_name}"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    logger.info(f"Push output: {push_result.stdout}")

                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed: {e.cmd}")
                    logger.error(f"Output: {e.output}")
                    logger.error(f"Error: {e.stderr}")
                    raise
                finally:
                    # Cleanup
                    if os.path.exists(modelfile_path):
                        os.unlink(modelfile_path)

                logger.info(f"Successfully pushed {base_name}:{tag_name}")

        except Exception as e:
            logger.error(f"Error pushing to Ollama: {str(e)}")
            raise


# Update main function call
@app.local_entrypoint()
def main(
    modelowner: str,
    modelname: str,
    username: str,
    quanttypes: Optional[str] = None,
    branch: Optional[str] = "",
    filter_path: Optional[str] = "",
    private: bool = False,
    ollama_upload: bool = False,
    hf_upload: bool = False,
    clean_run: bool = False,  
):
    logger.info(f"Starting conversion process for {modelowner}/{modelname}")
    converter = ModelConverter()

    try:
        model_id = f"{modelowner}/{modelname}"

        # Parse quanttypes from comma-separated string if provided
        quant_list = None
        if quanttypes:
            quant_list = [qt.strip() for qt in quanttypes.split(",")]

        converter.download_model.remote(
            model_id,
            modelname,
            username,
            branch,
            filter_path,
            quant_list or SUPPORTED_QUANT_TYPES,
            private,
            ollama_upload,  # Pass the Ollama upload flag
            hf_upload,  # Pass new parameter
            clean_run,  # Pass through
        )
        logger.info("Conversion process completed successfully")
    except Exception as e:
        logger.error(f"Conversion process failed: {str(e)}")
        raise
