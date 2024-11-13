# This code is a modified version of https://huggingface.co/spaces/ggml-org/gguf-my-repo/ from the Hugging Face to run in CLI instead Spaces, thank you for all your hard work 🙏🏼
import os
import logging
from modal import Image, Volume, Secret, App, method

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = App("hf-to-gguf-converter")

# Create Modal volume
volume = Volume.from_name("model-storage", create_if_missing=True)

# Base image with required dependencies
image = (
    Image.debian_slim()
    .apt_install("git", "curl", "build-essential", "cmake")
    .pip_install(
        "huggingface_hub",
        "torch",
        "transformers",
        "sentencepiece",
        "accelerate",
        "tqdm"
    )
    .run_commands(
        # Download and install HF downloader in two steps
        "curl -sSL https://g.bodaay.io/hfd > /tmp/hfd.sh",
        "bash /tmp/hfd.sh -i",
        # Clone and build llama.cpp
        "cd /root && git clone https://github.com/ggerganov/llama.cpp.git",
        "cd /root/llama.cpp && make",
        "cd /root/llama.cpp && pip install -r requirements.txt"
    )
)

@app.cls(
    image=image,
    secrets=[Secret.from_name("my-huggingface-secret")],
    volumes={"/root/models": volume},
    timeout=86400,
    gpu="A10G"
)
class ModelConverter:
    @method()
    def download_model(self, model_id: str):
        logger.info(f"Downloading model {model_id}...")
        import subprocess
        import time
        import re
        from datetime import datetime, timedelta

        try:
            local_dir = f"/root/models/{model_id.split('/')[-1]}-hf"
            os.makedirs(local_dir, exist_ok=True)

            # Track download progress
            last_progress = 0
            last_time = time.time()
            progress_pattern = r"Speed: ([\d.]+) MB/sec, ([\d.]+)%"

            cmd = [
                "hfdownloader",
                "-m", model_id,
                "-s", local_dir,
                "-c", "8",
                "-t", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    # Extract speed and progress
                    match = re.search(progress_pattern, output)
                    if match:
                        speed = float(match.group(1))
                        progress = float(match.group(2))
                        
                        # Calculate ETA
                        if progress > last_progress:
                            time_diff = time.time() - last_time
                            progress_diff = progress - last_progress
                            time_per_percent = time_diff / progress_diff
                            remaining_percent = 100 - progress
                            eta_seconds = time_per_percent * remaining_percent
                            eta = datetime.now() + timedelta(seconds=eta_seconds)
                            
                            # Update tracking variables
                            last_progress = progress
                            last_time = time.time()
                            
                            # Log with ETA
                            logger.info(f"{output.strip()} | ETA: {eta.strftime('%H:%M:%S')}")
                        else:
                            logger.info(output.strip())
                    else:
                        logger.info(output.strip())

                time.sleep(10)

            if process.returncode != 0:
                raise Exception(f"Download failed with exit code {process.returncode}")

            logger.info(f"Model downloaded to {local_dir}")
            # Commit volume after download
            volume.commit()
            logger.info("Volume committed after download")
            return local_dir

        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

    @method()
    def convert_to_gguf(self, input_dir: str, model_name: str, quant_type: str = "Q4_K_M"):
        logger.info(f"Converting model to GGUF format with quantization {quant_type}...")
        
        try:
            # Find the actual model directory containing config.json
            model_subdir = None
            for root, dirs, files in os.walk(input_dir):
                if 'config.json' in files:
                    model_subdir = root
                    break
                    
            if not model_subdir:
                raise Exception("Could not find config.json in model directory")
                
            logger.info(f"Found model files in: {model_subdir}")
            
            # First convert to f16
            fp16_path = f"/root/models/{model_name}.f16.gguf"
            logger.info("Converting to fp16 first...")
            
            cmd = f"python /root/llama.cpp/convert_hf_to_gguf.py {model_subdir} --outfile {fp16_path} --outtype f16"
            result = os.system(cmd)
            if result != 0:
                raise Exception(f"FP16 conversion failed with code {result}")
                
            # Then quantize
            output_file = f"/root/models/{model_name}.{quant_type.lower()}.gguf"
            logger.info(f"Quantizing to {quant_type}...")
            
            cmd = f"/root/llama.cpp/llama-quantize {fp16_path} {output_file} {quant_type}"
            result = os.system(cmd)
            if result != 0:
                raise Exception(f"Quantization failed with code {result}")
                
            logger.info(f"Model converted and quantized to {output_file}")
            volume.commit()
            return output_file
            
        except Exception as e:
            logger.error(f"Error in conversion/quantization: {str(e)}")
            raise

    @method()
    def split_model(self, model_path: str, max_tensors: int = 256):
        """Split large models into shards"""
        try:
            base_path = model_path.rsplit('.', 1)[0]
            cmd = f"/root/llama.cpp/llama-gguf-split --split --split-max-tensors {max_tensors} {model_path} {base_path}"
            
            result = os.system(cmd)
            if result != 0:
                raise Exception(f"Model splitting failed with code {result}")
                
            # Get list of generated shards
            shards = [f for f in os.listdir(os.path.dirname(model_path)) 
                     if f.startswith(os.path.basename(base_path))]
            
            return shards
        except Exception as e:
            logger.error(f"Error splitting model: {str(e)}")
            raise

    @method()
    def upload_to_hf(self, file_path: str, repo_id: str, source_model_id: str, quant_type: str, private: bool = False):
        logger.info("Reloading volume before upload...")
        volume.reload()
        
        logger.info(f"Uploading GGUF model to HuggingFace repo {repo_id}...")
        from huggingface_hub import HfApi, ModelCard
        from textwrap import dedent
        
        try:
            api = HfApi()
            
            # Create repo first
            api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
            
            # Generate model card
            try:
                card = ModelCard.load(source_model_id)
            except Exception:
                card = ModelCard("")
                
            if card.data.tags is None:
                card.data.tags = []
            card.data.tags.extend(["llama-cpp", "gguf"])
            card.data.base_model = source_model_id
            
            filename = os.path.basename(file_path)
            card.text = dedent(
                f"""
                # {repo_id}
                This model was converted to GGUF format from [`{source_model_id}`](https://huggingface.co/{source_model_id}) using llama.cpp.
                Refer to the [original model card](https://huggingface.co/{source_model_id}) for more details on the model.
                
                ## Use with llama.cpp
                
                ### CLI:
                ```bash
                llama-cli --hf-repo {repo_id} --hf-file {filename} -p "Your prompt here"
                ```
                
                ### Server:
                ```bash
                llama-server --hf-repo {repo_id} --hf-file {filename} -c 2048
                ```
                
                ## Model Details
                - **Quantization Type:** {quant_type}
                - **Original Model:** [{source_model_id}](https://huggingface.co/{source_model_id})
                - **Format:** GGUF
                """
            )
            
            # Save and upload README
            readme_path = "/tmp/README.md"
            card.save(readme_path)
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id
            )
            
            # Upload the model file
            logger.info(f"Uploading quantized model: {file_path}")
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

@app.local_entrypoint()
def main(modelowner: str, modelname: str, username: str, quanttype: str = "q8_0", private: bool = False):
    logger.info(f"Starting conversion process for {modelowner}/{modelname}")
    converter = ModelConverter()
    
    try:
        # Build proper model IDs
        model_id = f"{modelowner}/{modelname}"
        output_repo = f"{username}/{modelname}-{quanttype}-gguf"
        
        # Run conversion pipeline
        local_dir = converter.download_model.remote(model_id)
        gguf_path = converter.convert_to_gguf.remote(
            local_dir,
            model_name=modelname,
            quant_type=quanttype    
        )
        converter.upload_to_hf.remote(
            gguf_path,
            output_repo,
            source_model_id=model_id,
            quant_type=quanttype,
            private=private
        )
        
        logger.info("Conversion process completed successfully")
    except Exception as e:
        logger.error(f"Conversion process failed: {str(e)}")
        raise