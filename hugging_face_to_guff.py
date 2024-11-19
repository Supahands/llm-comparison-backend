# This code is a modified version of https://huggingface.co/spaces/ggml-org/gguf-my-repo/ from the Hugging Face to run in CLI instead Spaces, thank you for all your hard work 🙏🏼
import os
import logging
from modal import Image, Volume, Secret, App, method
from typing import Union, List, Optional

# Add at top of file
SUPPORTED_QUANT_TYPES = [
    "q2_k", 
    "q3_k_m", 
    "q4_0", 
    "q4_k_m",
    "q5_0", 
    "q5_k_m", 
    "q6_k",
    "q8_0"
]

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
    def download_model(self, model_id: str, modelname: str, username: str, branch: str = "", filter_path: str = "", quanttypes: Union[str, List[str]] = "q8_0", private: bool = False):
        logger.info(f"Downloading model {model_id}...")
        import subprocess
        import time
        import re
        from datetime import datetime, timedelta

        try:
            local_dir = f"/root/models/{model_id.split('/')[-1]}-hf"
            os.makedirs(local_dir, exist_ok=True)

            cmd = [
                "hfdownloader",
                "-m", model_id,
                "-s", local_dir,
                "-k",       
                "-c", "8",
            ]

            # Add branch if specified
            if branch:
                cmd.extend(["-b", branch])
                
            # Add filter if specified
            if filter_path:
                cmd.extend(["-f", filter_path])

            # Add token at the end
            cmd.extend(["-t", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")])

            # Track download progress
            last_progress = 0
            last_time = time.time()
            progress_pattern = r"Speed: ([\d.]+) MB/sec, ([\d.]+)%"

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
            
            # Call convert_to_gguf with .remote()
            return self.convert_to_gguf.remote(
                local_dir, 
                modelname,
                quanttypes,
                model_id,
                username,
                private,
                branch,  # Pass branch name to convert_to_gguf
                filter_path  # Pass filter_path to convert_to_gguf
            )

        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

    @method()
    def convert_to_gguf(self, input_dir: str, modelname: str, quanttypes: Union[str, List[str]], source_model_id: str, username: str, private: bool, branch: str = "", filter_path: str = ""):
        """Convert model to GGUF format with multiple quantization types"""
        logger.info(f"Converting model with quantization types: {quanttypes}")
        
        # Create repo name based on modelname, branch and filter if specified
        repo_base_name = modelname
        if branch:
            repo_base_name = f"{repo_base_name}-{branch}"
        if filter_path:
            clean_filter = filter_path.replace('/', '-').replace('_', '-').strip('-')
            repo_base_name = f"{repo_base_name}-{clean_filter}"
        
        # Create single repository name for all quantization versions
        output_repo = f"{username}/{repo_base_name}-gguf"
        
        if isinstance(quanttypes, str):
            quanttypes = [quanttypes]
        
        if not quanttypes:
            quanttypes = SUPPORTED_QUANT_TYPES
            
        try:
            # Find model directory
            model_subdir = None
            for root, dirs, files in os.walk(input_dir):
                if 'config.json' in files:
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
        
            # Upload all versions to the same repository
            return self.upload_to_hf.remote(
                model_files,
                output_repo,
                source_model_id,
                private
            )
                
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
            card.data.tags.extend(["llama-cpp", "gguf"])
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

# Update main function call
@app.local_entrypoint()
def main(
    modelowner: str, 
    modelname: str, 
    username: str, 
    quanttypes: Optional[str] = None,
    branch: Optional[str] = "",
    filter_path: Optional[str] = "",
    private: bool = False
):
    logger.info(f"Starting conversion process for {modelowner}/{modelname}")
    converter = ModelConverter()
    
    try:
        model_id = f"{modelowner}/{modelname}"
        
        # Parse quanttypes from comma-separated string if provided
        quant_list = None
        if (quanttypes):
            quant_list = [qt.strip() for qt in quanttypes.split(',')]
            
        converter.download_model.remote(
            model_id,
            modelname,
            username,
            branch,
            filter_path,
            quant_list or SUPPORTED_QUANT_TYPES,
            private
        )
        logger.info("Conversion process completed successfully")
    except Exception as e:
        logger.error(f"Conversion process failed: {str(e)}")
        raise