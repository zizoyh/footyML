import cv2
import os

input_dir = "data/raw_clips"
output_dir = "data/images/all_frames"
os.makedirs(output_dir, exist_ok=True)

frame_interval = 30  # 1 frame every ~1 sec (at 30 FPS)

for fname in os.listdir(input_dir):
    if not fname.endswith(".mp4"):
        continue

    cap = cv2.VideoCapture(os.path.join(input_dir, fname))
    basename = os.path.splitext(fname)[0]
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_dir, f"{basename}_f{frame_count}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"{fname}: saved {saved_count} frames.")
