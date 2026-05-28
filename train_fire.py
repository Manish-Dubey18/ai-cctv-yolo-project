# ============================================================
# train_fire.py — Train YOLOv8 fire detection model
# ============================================================
from ultralytics import YOLO

def train_fire_model():
    model = YOLO("yolov8n.pt")  # pretrained base

    results = model.train(
        data="dataset/fire.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        cos_lr=True,
        val=True,
        plots=True,
        save=True,
        project="runs/fire",
        name="fire_v1"
    )

    metrics = model.val()
    print(f"\n✅ Fire Model Results:")
    print(f"   mAP50     : {metrics.box.map50:.3f}")
    print(f"   mAP50-95  : {metrics.box.map:.3f}")
    print(f"   Precision : {metrics.box.mp:.3f}")
    print(f"   Recall    : {metrics.box.mr:.3f}")

    model.export(format="onnx")
    print("✅ Model exported to ONNX")

if __name__ == "__main__":
    train_fire_model()