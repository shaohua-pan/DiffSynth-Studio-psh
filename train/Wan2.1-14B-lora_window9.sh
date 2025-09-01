# accelerate launch examples/wanvideo/model_training/train.py \
#   --dataset_base_path data/example_video_dataset \
#   --dataset_metadata_path data/example_video_dataset/metadata.csv \
#   --height 480 \
#   --width 832 \
#   --dataset_repeat 100 \
#   --model_id_with_origin_paths "PAI/Wan2.1-Fun-1.3B-InP:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-1.3B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-1.3B-InP:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-1.3B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Wan2.1-Fun-1.3B-InP_lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
#   --lora_rank 32 \
#   --extra_inputs "input_image,end_image"

  # CUDA_VISIBLE_DEVICES=2 accelerate launch examples/wanvideo/model_training/train2.py \
#   --dataset_base_path data/example_video_dataset \
#   --dataset_metadata_path data/example_video_dataset/metadata.csv \
#   --height 480 \
#   --width 832 \
#   --dataset_repeat 100 \
#   --model_paths '[
#     "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
#     "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
#     "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
#     "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]' \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Wan2.1-Fun-1.3B-InP_full" \
#   --trainable_models "dit" \
#   --extra_inputs "input_image,end_image"
  # ps -ef|grep multi| awk '{print $2}'|xargs kill -9

  # accelerate launch --config_file single_machine_config.yaml train/train.py \
  # --dataset_base_path data/aimier \
  # --dataset_metadata_path data/aimier/metadata.csv \
  # --height 832 \
  # --width 480 \
  # --dataset_repeat 100000 \
  # --model_paths '[
  #   ["/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"],
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
  #   "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]' \
  # --learning_rate 1e-4  \
  # --num_epochs 5 \
  # --remove_prefix_in_ckpt "pipe.dit." \
  # --output_path "./models/train/Wan2.1-14B_lora_window9" \
  # --extra_inputs "input_image,use_latent_index" \
  # --train_framepack \
  # --wandb_api_key "f8f4c5655e1d92061c5f3f2da7c930fe4c14ef4d" \
  # --wandb_entity "baidu-vis" \
  # --wandb_project "wan-framepack" \
  # --save_steps 50 \
  # --latent_window_size 9 \
  # --lora_base_model "dit" \
  # --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  # --lora_rank 32 \
set -x
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
unset http_proxy
unset https_proxy
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=DEBUG
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
accelerate launch --debug --config_file /root/paddlejob/workspace/env_run/multi_machine_config.yaml train/train.py \
  --dataset_base_path data/aimier \
  --dataset_metadata_path data/aimier/metadata.csv \
  --height 832 \
  --width 480 \
  --dataset_repeat 100000 \
  --model_paths '[
    ["/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"],
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    "/root/paddlejob/workspace/env_run/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]'  \
  --learning_rate 1e-4  \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-14B_lora_window9" \
  --extra_inputs "input_image,use_latent_index" \
  --train_framepack \
  --wandb_api_key "f8f4c5655e1d92061c5f3f2da7c930fe4c14ef4d" \
  --wandb_entity "baidu-vis" \
  --wandb_project "wan-framepack" \
  --save_steps 50 \
  --latent_window_size 9 \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
