import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new2 import WanVideoPipeline, ModelConfig, ModelManager
from modelscope import dataset_snapshot_download
import os 
from diffsynth.models.wan_video_dit import WanModel
import imageio


def read_video_as_images(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        pil_img = Image.fromarray(frame)
        frames.append(pil_img)
    reader.close()
    return frames

videos = []
for i in range(6):
    prev_frames = read_video_as_images(f"output/yishushi_1frame_{i}.mp4")[1:]
    videos.extend(prev_frames)
save_video(videos, f"output/yishushi_1frame_all.mp4", fps=15, quality=5)


