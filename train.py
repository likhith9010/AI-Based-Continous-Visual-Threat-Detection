from ultralytics import YOLO

# Load the base model (pretrained on COCO — we fine-tune on top of this)
model = YOLO("yolov8n.pt")

# Start training
results = model.train(
    data="datasets/weapon_detection/data.yaml",  # path to your dataset config
    epochs=50,                        # 50 passes through the data
    imgsz=640,                        # standard image size
    batch=8,                          # safe for RTX 3050 (4GB VRAM)
    device="cpu",                     # CPU for now — see below to enable GPU
    patience=10,                      # stop early if no improvement
    project="runs/train",             # where to save results
    name="guns_model",                # folder name for this run
    verbose=True
)

print("\n✅ Training complete!")
print(f"Best model saved at: runs/train/guns_model/weights/best.pt")
print("\nNext step: update gui.py line 64 to use your new model:")
print('  self.yolo_model = YOLO("runs/train/guns_model/weights/best.pt")')
