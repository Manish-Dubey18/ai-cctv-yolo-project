import numpy as np
import cv2

class AnomalyDetector:
    def __init__(self, model_path="models/anomaly_model.h5", sequence_length=16, threshold=0.6):
        """
        Initialize Anomaly Detector using sequence-based model.
        """
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.frame_buffer = []
        self.img_size = (64, 64)
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"[AnomalyDetector] Loaded model: {model_path}")
        except Exception as e:
            print(f"[AnomalyDetector] Model not loaded: {e}. Running in demo mode.")
            self.model = None

    def _preprocess_frame(self, frame):
        resized = cv2.resize(frame, self.img_size)
        normalized = resized / 255.0
        return normalized

    def detect(self, frame):
        """
        Detect anomalies in a video sequence.
        Returns: (label: str, score: float, anomaly_detected: bool)
        """
        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)

        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)

        if self.model is None or len(self.frame_buffer) < self.sequence_length:
            return "Analyzing...", 0.0, False

        sequence = np.expand_dims(np.array(self.frame_buffer), axis=0)

        try:
            score = float(self.model.predict(sequence, verbose=0)[0][0])
            anomaly_detected = score > self.threshold
            label = "Anomaly Detected" if anomaly_detected else "Normal"
            return label, score, anomaly_detected
        except Exception as e:
            return "Error", 0.0, False