import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new2 import WanVideoPipeline, ModelConfig, ModelManager
from modelscope import dataset_snapshot_download
import os 
from diffsynth.models.wan_video_dit import WanModel
import imageio

tokenizer_config = ModelConfig(path="/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-T2V-1.3B/google/umt5-xxl")

model_path = '/root/paddlejob/bosdata/huangluying/Wan/Wan2.1-FLF2V-14B-720P/'
dit_path = sorted([os.path.join(model_path, x) for x in os.listdir(model_path) if "diffusion_pytorch_model-0000" in x])
vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
text_encoder_path = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
image_encoder_path = os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        image_encoder_path,
        vae_path
    ],
    torch_dtype=torch.float32,
)

model_manager.load_models(
    [
        dit_path,
        text_encoder_path,
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)
model_manager.load_lora('/root/paddlejob/bosdata/wangkaisiyuan/NOVA/Model_Backup/20250702_manual/1800.bin', lora_alpha=1.0)

pipe = WanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device=f"cuda"
    )

pipe.enable_vram_management(num_persistent_param_in_dit=None)


def read_video_as_images(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        pil_img = Image.fromarray(frame)
        frames.append(pil_img)
    reader.close()
    return frames

image = Image.open("/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/yishushi.jpg")

videos = []
prev_frames = read_video_as_images('/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/output/yishushi1.mp4')[:41]
for i in range(48):
    video = pipe(
        prompt="主播正在专注地讲解商品，她的动作缓慢，手指清晰可见，动作幅度很小。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=prev_frames[0],
        end_image=prev_frames[-1],
        seed=0, tiled=True,
        height=832, width=480,
        prev_frames=prev_frames,
    )
    if len(videos) == 0:
        videos.extend(video)
    else:
        videos.extend(video[41:]) 
    save_video(video, f"output/yishushi_his_41frame_{i}.mp4", fps=15, quality=5)
    prev_frames = read_video_as_images(f"output/yishushi_his_41frame_{i}.mp4")[-41:]
save_video(videos, f"output/yishushi_his_41frame_all.mp4", fps=15, quality=5)