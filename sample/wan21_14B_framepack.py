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
pipe.load_lora(pipe.dit, '/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/models/train/Wan2.1-14B_lora_window9/step-1450.safetensors')
pipe.enable_vram_management()

reader = imageio.get_reader('/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/420_450.mp4')
input_image = Image.fromarray(reader.get_data(0))

import torch
from torchvision.transforms import ToTensor, ToPILImage


def soft_append_bcthw(history, current, overlap=0):
    if overlap <= 0:
        return torch.cat([history, current], dim=2)

    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
    
    weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)


def soft_append_videos(history_imgs, current_imgs, overlap=4):
    if not current_imgs:
        return history_imgs
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    # ⚠️ 统一尺寸
    # target_size = history_imgs[0].size
    # current_imgs = [img.resize(target_size, Image.BICUBIC) for img in current_imgs]

    history_tensor = torch.stack([to_tensor(img) for img in history_imgs]).unsqueeze(0).transpose(1, 2)
    current_tensor = torch.stack([to_tensor(img) for img in current_imgs]).unsqueeze(0).transpose(1, 2)
    print(history_tensor.shape, current_tensor.shape)
    output_tensor = soft_append_bcthw(history_tensor, current_tensor, overlap=overlap)
    output_imgs = [to_pil(output_tensor[0, :, t]) for t in range(output_tensor.shape[2])]
    return output_imgs



videos = []
total_second_length = 30
latent_window_size = 9
total_latent_sections = (total_second_length * 25) / (latent_window_size * 4) # seconds * fps / window_size * 4, VAE latent uses 4 frame as a patch
total_latent_sections = int(max(round(total_latent_sections), 1))
latent_paddings = reversed(range(total_latent_sections))
if total_latent_sections > 4:
    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
post_image, post_image2x, post_image4x = None, None, None
for idx, latent_padding in enumerate(latent_paddings):
    is_last_section = latent_padding == 0
    latent_padding_size = latent_padding * latent_window_size
    print(f'idx = {idx}, all_idx = {len(latent_paddings)}')
    print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')
    video = pipe(
        prompt="视频以一位身穿红色上衣、佩戴项链与耳环的女性为主角，背景是淡雅的山水画与飞鸟点缀的静谧场景，桌上整齐摆放着书脊带有金色文字的书籍。首先，她双手轻搭书上，右手抬起引导视线，随后握拳强调观点，左手随之触碰书本并同步做出握拳动作，头部微微前倾，展现出专注与投入。紧接着，她的手势继续变化，时而轻触书页，时而双手交叉于桌面，手掌朝下，表现出沉思或准备继续讲解的姿态。最终，她右手再次抬起，左手加入，双手交替做出强调动作，手指轻轻敲击书本，仿佛在总结或引出新话题。整个场景风格简洁，动作流畅自然，突出人物与书籍的互动，营造出专业而温馨的氛围，推测为知识讲解或产品介绍类内容。",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=input_image,
        seed=idx*100, tiled=True,
        height=832, width=480,
        num_frames=(latent_window_size + 1) * 4 + 1,
        use_latent_index=True,
        latent_padding=latent_padding_size,
        post_images=[post_image, post_image2x, post_image4x] if post_image is not None else None,
        latent_window_size=latent_window_size,
    )
    post_image, post_image2x, post_image4x = video[1:5], None, None
    # videos = video[1:-4] + videos
    videos = soft_append_videos(video[1:], videos, overlap=4)
    save_video(video, f"output/aimier_14B_framepack_{idx}.mp4", fps=25, quality=5)
    # save_video(video[-4:], f"output/aimier_14B_framepack_-4_{idx}.mp4", fps=15, quality=5)
    # save_video(video[1:5], f"output/aimier_14B_framepack_1-5_{idx}.mp4", fps=15, quality=5)
    if is_last_section:
        videos = [input_image.resize((480, 832), Image.BICUBIC)] + videos
        save_video(videos, f"output/aimier_14B_framepack_all.mp4", fps=25, quality=5)
        break


# import imageio
# from PIL import Image
# import numpy as np

# def resize_frame(frame, target_size):
#     """统一尺寸到 target_size"""
#     img = Image.fromarray(frame)
#     if img.size != target_size:
#         img = img.resize(target_size, Image.BICUBIC)
#     return np.array(img)

# # 小视频路径（12 段）
# video_paths = [f"output/aimier_14B_framepack_{i}.mp4" for i in range(12)]

# # 确定统一尺寸：取最后一个视频的第一帧（和训练时一致）
# reader0 = imageio.get_reader(video_paths[-1])
# target_frame = reader0.get_data(0)
# target_size = Image.fromarray(target_frame).size  # (W, H)
# reader0.close()

# all_frames = []

# # 遍历每个小视频，按逻辑拼接
# for path in video_paths:
#     reader = imageio.get_reader(path)
#     frames = [resize_frame(frame, target_size) for frame in reader]
#     reader.close()

#     # 模拟 video[1:-4] + videos 逻辑
#     if len(all_frames) == 0:
#         all_frames = frames[1:] + all_frames
#     else:
#         all_frames = soft_append_videos(frames[1:], all_frames, overlap=4)



# # ⚠️ 注意：如果 input_image 需要是视频第一帧，应该用 imageio 读取
# reader_in = imageio.get_reader("/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/samples/420_450.mp4")
# first_frame = reader_in.get_data(0)
# reader_in.close()

# first_frame_resized = resize_frame(first_frame, target_size)
# all_frames = [first_frame_resized] + all_frames

# # 写出拼接后的视频
# writer = imageio.get_writer("output/aimier_14B_framepack_all.mp4", fps=25, quality=8)
# for f in all_frames:
#     if isinstance(f, Image.Image):  # 如果是 PIL.Image，转成 ndarray
#         f = np.array(f)
#     writer.append_data(f)
# writer.close()

# print("✅ 拼接完成：output/aimier_14B_framepack_all.mp4")
