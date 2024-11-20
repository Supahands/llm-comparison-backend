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
        hf_upload: bool = False,  # New parameter
    ):
        logger.info(f"Downloading model {model_id}...")
        import subprocess

        try:
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
                hf_upload,  # Pass new parameter
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
        hf_upload: bool = False,  # New parameter
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
                    model_files, modelname, source_model_id, username
                )

            # You can also upload to Hugging Face if needed
            if hf_upload:
                return self.upload_to_hf.remote(
                    model_files,
                    f"{username}/{modelname}-gguf",
                    source_model_id,
                    private
                )

        except Exception as e:
            logger.error(f"Error in conversion/quantization: {str(e)}")
            raise

    @method()
    def push_to_ollama(
        self,
        model_files: List[tuple],
        modelname: str,
        source_model_id: str,
        username: str,
    ):
        """Push converted models to Ollama using tags for different quantizations"""
        logger.info("Pushing models to Ollama...")

        try:
            base_name = f"{username}/{modelname}"

            for model_path, quant_type in model_files:
                tag_name = quant_type.lower()

                # Create Modelfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.Modelfile', delete=False) as f:
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
                    logger.info(f"Created Modelfile for {base_name}:{tag_name}:\n{modelfile_content}")

                # Create model in Ollama with tag
                logger.info(f"Creating Ollama model {base_name}:{tag_name}")
                create_cmd = ["ollama", "create", f"{base_name}:{tag_name}", "-f", f.name]
                subprocess.run(create_cmd, check=True)

                # Push to Ollama registry
                logger.info(f"Pushing model {base_name}:{tag_name} to Ollama")
                push_cmd = ["ollama", "push", f"{base_name}:{tag_name}"]
                subprocess.run(push_cmd, check=True)

                logger.info(f"Successfully pushed {base_name}:{tag_name} to Ollama")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error pushing to Ollama: {e}")
            raise Exception(f"Failed to push Ollama model: {base_name}:{tag_name}") from e
        except Exception as e:
            logger.error(f"Error pushing to Ollama: {e}")
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
    hf_upload: bool = False,  # New parameter
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
        )
        logger.info("Conversion process completed successfully")
    except Exception as e:
        logger.error(f"Conversion process failed: {str(e)}")
        raise
