import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_framepack import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import imageio
from PIL import Image

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
state_dict = load_state_dict("/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/ckpts/Wan2.1-Fun-1.3B-InP_full/step-20.safetensors")
pipe.dit.load_state_dict(state_dict)
pipe.enable_vram_management()

reader = imageio.get_reader('/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/420_450.mp4')
input_image = Image.fromarray(reader.get_data(0))

videos = []
total_second_length = 30
latent_window_size = 9
total_latent_sections = (total_second_length * 15) / (latent_window_size * 4) # seconds * fps / window_size * 4, VAE latent uses 4 frame as a patch
total_latent_sections = int(max(round(total_latent_sections), 1))
latent_paddings = reversed(range(total_latent_sections))
if total_latent_sections > 4:
    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
for idx, latent_padding in enumerate(latent_paddings):
    is_last_section = latent_padding == 0
    latent_padding_size = latent_padding * latent_window_size
    print(f'idx = {idx}, all_idx = {len(latent_paddings)}')
    print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')
    post_image, post_image2x, post_image4x = None, None, None
    video = pipe(
        prompt="视频以一位身穿红色上衣、佩戴项链与耳环的女性为主角，背景是淡雅的山水画与飞鸟点缀的静谧场景，桌上整齐摆放着书脊带有金色文字的书籍。首先，她双手轻搭书上，右手抬起引导视线，随后握拳强调观点，左手随之触碰书本并同步做出握拳动作，头部微微前倾，展现出专注与投入。紧接着，她的手势继续变化，时而轻触书页，时而双手交叉于桌面，手掌朝下，表现出沉思或准备继续讲解的姿态。最终，她右手再次抬起，左手加入，双手交替做出强调动作，手指轻轻敲击书本，仿佛在总结或引出新话题。整个场景风格简洁，动作流畅自然，突出人物与书籍的互动，营造出专业而温馨的氛围，推测为知识讲解或产品介绍类内容。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=input_image,
        seed=23333, tiled=True,
        height=1920//2, width=1080//2,
        num_frames=(9 + 1 + 1 + 1) * 4 + 1,
        use_latent_index=True,
        latent_padding=latent_padding_size,
        post_images=[post_image, post_image2x, post_image4x] if post_image is not None else None
    )
    if is_last_section:
        save_video(videos, f"output/yishushi_1frame_1.3B_framepack_all.mp4", fps=15, quality=5)
        break
    post_image, post_image2x, post_image4x = video[0], video[1], video[2]
    overlapped_frames = latent_window_size * 4 - 3
    videos = video[:-3] + videos
    save_video(video, f"output/yishushi_1frame_1.3B_framepack_{idx}.mp4", fps=15, quality=5)
