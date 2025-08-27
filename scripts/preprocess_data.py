import json
import csv
import os
import imageio

# 输入文件路径
input_file = "/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/aimier/human_videos_30s_merge_caption"   # 你的文件路径
output_file = "/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/data/aimier/aimier_metadata.csv"

# 打开输出 CSV 文件
with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video", "prompt"])  # 写表头

    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                crop_bos_path = data.get("crop_bos_path", "")
                meta = data.get("meta", {})
                prompt = meta.get("describe_merge_72b", "")

                # 判断是否包含 "艾弥尔"
                if "艾弥尔" in crop_bos_path:
                    # 获取视频文件名
                    video_name = os.path.join("/root/paddlejob/bosdata/", crop_bos_path)
                    try:
                        reader = imageio.get_reader(video_name)
                        # 尝试读取第一帧验证是否正常
                        _ = reader.get_data(0)
                        reader.close()
                    except Exception as e:
                        print(f"跳过无法读取的视频: {video_name}, 错误: {e}")
                        continue  # 跳过这条数据
                    writer.writerow([video_name, prompt])
            except json.JSONDecodeError:
                print(f"跳过无法解析的行: {line}")

print(f"处理完成，结果已保存到 {output_file}")
