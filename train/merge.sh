CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path /workspace/user_code/cfs-qndkq39v/zejingxie/hug_weight/Qwen2.5-VL-7B-Instruct \
    --adapter_name_or_path ./saves/Qwen2.5-VL-7B-Instruct/lora/train_2025-06-18  \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_dir /workspace/user_code/cfs-qndkq39v/zejingxie/hug_weight/Qwen2.5-VL-7B-Instruct_lora \
    --export_size 2 \
    --export_device auto \
    --export_legacy_format False