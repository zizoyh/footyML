import cv2
import numpy as np

class BallTracker:
    def __init__(self):
        # Initialize a 4D Kalman Filter (x, y, dx, dy)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        self.last_prediction = None
        self.frames_since_seen = 0
        self.max_invisible_frames = 15  # stop showing box after this many misses

    def update(self, detection=None):
        if detection is not None:
            x1, y1, x2, y2 = detection["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            self.kalman.correct(measurement)
            self.frames_since_seen = 0
        else:
            self.frames_since_seen += 1

        prediction = self.kalman.predict()
        self.last_prediction = prediction
        return prediction

    def get_tracked_bbox(self, box_width=20, box_height=20):
        if self.last_prediction is None or self.frames_since_seen > self.max_invisible_frames:
            return None

        cx = int(self.last_prediction[0])
        cy = int(self.last_prediction[1])
        x1 = cx - box_width // 2
        y1 = cy - box_height // 2
        x2 = cx + box_width // 2
        y2 = cy + box_height // 2
        return [x1, y1, x2, y2]

    def is_tracking(self):
        return self.frames_since_seen <= self.max_invisible_frames
