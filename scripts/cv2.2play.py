import cv2
import numpy as np

def merge_videos(video1_path, video2_path, start_frame1=0, start_frame2=0, output_path='output.mp4', fps=None):
    # 打开视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 跳到指定帧
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame1)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame2)

    # 视频尺寸
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 拼接视频尺寸（横向拼接）
    out_w = w1 + w2
    out_h = max(h1, h2)

    # FPS
    if fps is None:
        fps = cap1.get(cv2.CAP_PROP_FPS)

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # 如果两视频高度不同，调整高度一致
        if h1 != h2:
            frame2 = cv2.resize(frame2, (w2, h1))

        # 拼接
        combined = np.hstack((frame1, frame2))

        # 计算像素平方差
        min_h = min(frame1.shape[0], frame2.shape[0])
        min_w = min(frame1.shape[1], frame2.shape[1])
        diff = frame1[:min_h, :min_w].astype(np.float32) - frame2[:min_h, :min_w].astype(np.float32)
        ssd = np.sum(diff ** 2)

        # 在左上角显示SSD
        text = f'SSD: {int(ssd)}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6  # 稍大一点
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        # 背景矩形
        cv2.rectangle(combined, (5,5), (10+text_w, 10+text_h), (0,0,0), -1)  # 黑色填充
        cv2.putText(combined, text, (10, 10+text_h-2), font, font_scale, (0,0,255), thickness)

        # 写入视频
        out.write(combined)

    cap1.release()
    cap2.release()
    out.release()
    print("视频合成完成，保存为:", output_path)


if __name__ == "__main__":
    merge_videos('video11.mp4', 'video22.mp4', start_frame1=81-36+1, start_frame2=0, output_path='merged.mp4')
