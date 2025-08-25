import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

tokenizer_config = ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-T2V-1.3B/google/umt5-xxl")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path='/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors', offload_device="cpu"),
        ModelConfig(path='/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth', offload_device="cpu"),
        ModelConfig(path='/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', offload_device="cpu"),
        ModelConfig(path='/root/paddlejob/workspace/env_run/panshaohua/DiffSynth-Studio-psh/test/models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', offload_device="cpu"),
    ],
    tokenizer_config=tokenizer_config
)
pipe.enable_vram_management()

videos = []
image = Image.open("/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/yishushi.jpg")
prev_frames = [image for i in range(40)]
for i in range(6):
    video = pipe(
        prompt="主播正在专注地讲解商品，她的动作缓慢，手指清晰可见，动作幅度很小。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=prev_frames[-1],
        seed=23333, tiled=True,
        height=832, width=480,
    )
    if len(videos) == 0:
        videos.extend(video)
    else:
        videos.extend(video[1:]) 
    save_video(video, f"output/yishushi_1frame_1.3B_{i}.mp4", fps=15, quality=5)
    prev_frames = video
save_video(videos, f"output/yishushi_1frame_1.3B_all.mp4", fps=15, quality=5)