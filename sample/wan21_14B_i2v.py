import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, ModelManager
from modelscope import dataset_snapshot_download
import os 
from diffsynth.models.wan_video_dit import WanModel
import imageio

tokenizer_config = ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-T2V-1.3B/google/umt5-xxl")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ], offload_device="cpu"),
        ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
    tokenizer_config=tokenizer_config,
)
pipe.enable_vram_management()
videos = []
image = Image.open("/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/dance.png")
prev_frames = [image for i in range(40)]
for i in range(12):
    video = pipe(
        prompt="The woman dances elegantly among the blossoms, spinning slowly with flowing sleeves and graceful hand movements.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=prev_frames[-1],
        seed=23333, tiled=True,
        height=768, width=512,
    )
    if len(videos) == 0:
        videos.extend(video)
    else:
        videos.extend(video[1:]) 
    save_video(video, f"output/1dance_1_-20frame_{i}.mp4", fps=15, quality=5)
    prev_frames = video
save_video(videos, f"output/1dance_1_-20frame_all.mp4", fps=15, quality=5)