# accelerate launch examples/wanvideo/model_training/train.py \
#   --dataset_base_path data/example_video_dataset \
#   --dataset_metadata_path data/example_video_dataset/metadata.csv \
#   --height 480 \
#   --width 832 \
#   --dataset_repeat 100 \
#   --model_id_with_origin_paths "PAI/Wan2.1-Fun-1.3B-InP:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-1.3B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-1.3B-InP:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-1.3B-InP:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Wan2.1-Fun-1.3B-InP_full" \
#   --trainable_models "dit" \
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


  accelerate launch train/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_paths '[
    "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
    "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    "/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]' \
  --learning_rate 1e-5 \
  --num_epochs 30 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./ckpts/Wan2.1-Fun-1.3B-InP_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,latent_index"
  # wandb log
  --wandb_api_key "f8f4c5655e1d92061c5f3f2da7c930fe4c14ef4d" \
  --wandb_entity "baidu-vis" \
  --wandb_project "wan-framepack" \