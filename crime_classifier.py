import numpy as np
import cv2

# Crime classes based on UCF Crime dataset
CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

class CrimeClassifier:
    def __init__(self, model_path="models/crime_model.h5", sequence_length=16):
        """
        Initialize Crime Classifier using a sequence-based deep learning model.
        """
        self.sequence_length = sequence_length
        self.frame_buffer = []
        self.img_size = (64, 64)
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"[CrimeClassifier] Loaded model: {model_path}")
        except Exception as e:
            print(f"[CrimeClassifier] Model not loaded: {e}. Running in demo mode.")
            self.model = None

    def _preprocess_frame(self, frame):
        resized = cv2.resize(frame, self.img_size)
        normalized = resized / 255.0
        return normalized

    def detect(self, frame):
        """
        Run crime classification on a frame using sequence buffer.
        Returns: (label: str, confidence: float, crime_detected: bool)
        """
        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)

        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)

        if self.model is None or len(self.frame_buffer) < self.sequence_length:
            return "Analyzing...", 0.0, False

        sequence = np.expand_dims(np.array(self.frame_buffer), axis=0)  # (1, T, H, W, C)

        try:
            predictions = self.model.predict(sequence, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            label = CRIME_CLASSES[class_idx] if class_idx < len(CRIME_CLASSES) else "Unknown"
            crime_detected = label != "Normal" and confidence > 0.6
            return label, confidence, crime_detected
        except Exception as e:
            return "Error", 0.0, False