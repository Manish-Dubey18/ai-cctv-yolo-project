import cv2
import numpy as np

class CrowdDetector:
    def __init__(self, model_path="models/crowd_model.h5", crowd_threshold=10):
        """
        Initialize Crowd Detector.
        crowd_threshold: number of people above which crowd alert triggers
        """
        self.crowd_threshold = crowd_threshold
        self.model = None
        self._load_model(model_path)

        # Fallback: use YOLO person detection if no crowd model
        self.yolo = None
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO("yolov8n.pt")
                print("[CrowdDetector] Using YOLOv8 for person detection")
            except Exception as e:
                print(f"[CrowdDetector] YOLO not available: {e}")

    def _load_model(self, model_path):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"[CrowdDetector] Loaded model: {model_path}")
        except Exception as e:
            print(f"[CrowdDetector] Model not loaded: {e}. Falling back to YOLO person detection.")

    def detect(self, frame):
        """
        Detect crowd in frame.
        Returns: (annotated_frame, count: int, crowd_alert: bool)
        """
        annotated = frame.copy()

        # Use YOLO fallback — count persons (class 0 in COCO)
        if self.yolo is not None:
            results = self.yolo(frame, conf=0.4, classes=[0], verbose=False)
            boxes = results[0].boxes
            count = len(boxes) if boxes is not None else 0
            annotated = results[0].plot()
        else:
            count = 0  # model not loaded

        crowd_alert = count >= self.crowd_threshold

        # Overlay count on frame
        color = (0, 0, 255) if crowd_alert else (0, 255, 0)
        cv2.putText(annotated, f"People: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        if crowd_alert:
            cv2.putText(annotated, "⚠ CROWD ALERT", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return annotated, count, crowd_alert