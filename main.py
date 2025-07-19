import cv2
import os
from models.ball_detector import BallDetector

# --- Configuration ---
VIDEO_PATH = "data/clip.mp4"  # Replace with your actual video file
MODEL_PATH = "models/yolov8_ball.pt"              # Replace with your trained YOLO model
OUTPUT_PATH = "output/annotated_ball_output.mp4"
CONFIDENCE_THRESHOLD = 0.25
DISPLAY_LIVE = True
SAVE_OUTPUT = True


def initialize_detector():
    """Initialize the ball detection model."""
    return BallDetector(model_path=MODEL_PATH)


def process_video(video_path, detector, conf_thresh, output_path=None):
    """
    Process the video frame by frame, run detection, and visualize results.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if SAVE_OUTPUT and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, conf_thresh=conf_thresh)
        annotated = draw_detections(frame.copy(), detections)

        if DISPLAY_LIVE:
            cv2.imshow("Ball Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
    if DISPLAY_LIVE:
        cv2.destroyAllWindows()


def draw_detections(frame, detections):
    """Draw bounding boxes on the detected ball(s)."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"Ball {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


if __name__ == "__main__":
    ball_detector = initialize_detector()
    process_video(VIDEO_PATH, ball_detector, CONFIDENCE_THRESHOLD, OUTPUT_PATH)
