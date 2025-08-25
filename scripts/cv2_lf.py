import cv2

def save_last_frame_fast(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # 跳到最后一帧
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print("最后一帧已保存到:", output_path)
    else:
        print("读取最后一帧失败")


if __name__ == "__main__":
    save_last_frame_fast("dance_2.1_1.mp4", "last_frame_dance_wan21_1.jpg")


