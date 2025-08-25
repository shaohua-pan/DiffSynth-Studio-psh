import cv2

# 输入视频路径
video1_path = "/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/output/yishushi1.mp4"
video2_path = "/root/paddlejob/workspace/env_run/panshaohua/code/DiffSynth-Studio-psh/output/yishushi_his_41frame_all1.mp4"
output_path = "output/output_concat.mp4"

# 第二个视频从第 n 帧开始
n = 50

# 打开视频
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# 获取参数
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)

# 如果 fps 不一致，用第一个视频的 fps
fps = int(fps1)

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 统一视频尺寸（取第一个视频的尺寸作为基准）
if (width1, height1) != (width2, height2):
    resize_needed = True
else:
    resize_needed = False

# 跳过第二个视频前 n 帧
for _ in range(n):
    ret = cap2.grab()
    if not ret:
        print("第二个视频长度不足，无法跳过这么多帧！")
        break

# 输出视频参数（按第一个视频的分辨率）
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width1, height1))

# 写入第一个视频
while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        break
    out.write(frame1)

# 写入第二个视频（从第 n 帧开始）
while True:
    ret2, frame2 = cap2.read()
    if not ret2:
        break
    if resize_needed:
        frame2 = cv2.resize(frame2, (width1, height1))
    out.write(frame2)

# 释放资源
cap1.release()
cap2.release()
out.release()
print(f"拼接完成，输出文件：{output_path}")
