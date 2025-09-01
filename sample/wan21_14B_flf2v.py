import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new2 import WanVideoPipeline, ModelConfig, ModelManager
from modelscope import dataset_snapshot_download
import os 
from diffsynth.models.wan_video_dit import WanModel
import imageio
from PIL import Image

tokenizer_config = ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-T2V-1.3B/google/umt5-xxl")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ], offload_device="cpu"),
        ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
    tokenizer_config=tokenizer_config,
)
pipe.load_lora(pipe.dit, '/root/paddlejob/bosdata/wangkaisiyuan/NOVA/Model_Backup/20250702_manual/1800.bin')    
pipe.enable_vram_management()

# pipe_i2v = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(path=[
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
#             "/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
#         ], offload_device="cpu"),
#         ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
#         ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth", offload_device="cpu"),
#         ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
#     ],
#     tokenizer_config=tokenizer_config,
# )
# pipe_i2v.enable_vram_management()

videos = []
# image = Image.open("/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/yishushi.jpg")
reader = imageio.get_reader('/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/20_30.mp4')
prev_frames = [Image.fromarray(reader.get_data(i)) for i in range(5)]
next_frames = [Image.fromarray(reader.get_data(i)) for i in range(100, 104)]
video = pipe(
            prompt="主播正在专注地讲解商品，她的动作缓慢，手指清晰可见，动作幅度很小。",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            prev_frames=prev_frames,
            next_frames=next_frames,
            input_image=prev_frames[0],
            end_image=next_frames[-1],
            seed=23333, tiled=True,
            height=832, width=480,
            cfg_scale=5.5, sigma_shift=16.0,
        )
videos.extend(video)
for i in range(1, 5):
    prev_frames = videos[i*7+(i-1)*79:i*7+5+(i-1)*79]
    next_frames = videos[i*7+5+(i-1)*79:i*7+9+(i-1)*79]
    video = pipe(
        prompt="主播正在专注地讲解商品，她的动作缓慢，手指清晰可见，动作幅度很小。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        prev_frames=prev_frames,
        next_frames=next_frames,
        input_image=prev_frames[0],
        end_image=next_frames[-1],
        seed=23333+i, tiled=True,
        height=832, width=480,
        cfg_scale=5.5, sigma_shift=16.0,
    )
    videos = videos[:i*7+(i-1)*79] + video + videos[i*7+9+(i-1)*79:]
    save_video(video, f"output/yishushi_flf2v_lora2_{i}.mp4", fps=15, quality=5)
save_video(videos, f"output/yishushi_flf2v_lora2__all.mp4", fps=15, quality=5)