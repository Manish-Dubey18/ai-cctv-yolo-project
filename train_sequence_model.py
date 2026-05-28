# ============================================================
# train_sequence_models.py — Train Crime & Anomaly models
# ============================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import os

SEQUENCE_LENGTH = 16
IMG_SIZE = 64
CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

def build_lstm_model(num_classes):
    """Build ConvLSTM sequence classification model."""
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(128, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_anomaly_model():
    """Build binary anomaly detection model."""
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.4),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # binary: normal vs anomaly
    ])
    return model

def train_crime_model():
    print("📦 Loading crime sequences...")
    X = np.load("dataset/X_sequences.npy")
    y = np.load("dataset/y_sequences.npy")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model(num_classes=len(CRIME_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/crime_model.h5", save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=8, callbacks=cb)

    loss, acc = model.evaluate(X_val, y_val)
    print(f"\n✅ Crime Model — Val Accuracy: {acc:.3f}")

def train_anomaly_model():
    print("📦 Loading anomaly sequences...")
    X = np.load("dataset/X_sequences.npy")
    y = np.load("dataset/y_sequences.npy")

    # Binary: 0 = Normal (class index 7), 1 = Anomaly
    y_binary = (y != 7).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    model = build_anomaly_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/anomaly_model.h5", save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=8, callbacks=cb)

    results = model.evaluate(X_val, y_val)
    print(f"\n✅ Anomaly Model — Val Accuracy: {results[1]:.3f}, AUC: {results[2]:.3f}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_crime_model()
    train_anomaly_model()