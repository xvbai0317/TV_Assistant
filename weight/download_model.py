from huggingface_hub import snapshot_download

# Model name on Hugging Face
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# Local directory to save the model
local_dir = "/root/TV_Assistant/weight/Qwen"

print(f"Downloading {model_name} to {local_dir}...")

# Download the model files
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=["*.bin", "*.json", "*.model", "*.safetensors", "*.txt"]
)

print("Download completed!")