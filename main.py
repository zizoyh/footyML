import cv2
import os
from models.ball_detector import BallDetector
from models.ball_tracker import BallTracker

# --- Configuration ---
VIDEO_PATH = "data/raw_clips/clip2.mp4"
MODEL_PATH = "runs/detect/train9/weights/best.pt"
OUTPUT_PATH = "output/annotated_ball_output.mp4"
CONFIDENCE_THRESHOLD = 0.70
DISPLAY_LIVE = True
SAVE_OUTPUT = False


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

    tracker = BallTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, conf_thresh=conf_thresh)
        ball_det = detections[0] if detections else None

        tracker.update(ball_det)
        tracked_box = tracker.get_tracked_bbox()

        annotated = frame.copy()
        if tracked_box:
            x1, y1, x2, y2 = tracked_box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, "Ball (tracked)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if DISPLAY_LIVE:
            cv2.imshow("Ball Detection + Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
    if DISPLAY_LIVE:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ball_detector = initialize_detector()
    process_video(VIDEO_PATH, ball_detector, CONFIDENCE_THRESHOLD, OUTPUT_PATH)