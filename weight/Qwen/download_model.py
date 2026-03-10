from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/root/autodl-tmp/cache')
print(model_dir)