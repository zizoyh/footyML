import cv2
import torch
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class BallDetector:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load the ball detection model from disk.
        You can replace this with a specific YOLOv5/YOLOv8 loading logic.
        """
        # Example with YOLOv5 from ultralytics
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.fuse()
            return model
        except ImportError as e:
            raise ImportError("Install ultralytics to use YOLO-based detection.") from e

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.25):
        """
        Detect the ball in a given frame.

        Args:
            frame (np.ndarray): BGR image frame from video.
            conf_thresh (float): Minimum confidence threshold.

        Returns:
            List of dicts with bounding box and confidence:
                [{"bbox": [x1, y1, x2, y2], "conf": 0.85}, ...]
        """
        results = self.model(frame, verbose=False)[0]
        ball_detections = []

        for result in results.boxes:
            cls_id = int(result.cls.item())
            conf = result.conf.item()
            if conf < conf_thresh:
                continue
            
            # Assume class 0 is ball — change as needed
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                ball_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf
                })

        return ball_detections
