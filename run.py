import os
from ultralytics import YOLO

def main():
    # The path where YOLO saves the last checkpoint
    checkpoint_path = "runs/train/guns_model/weights/last.pt"
    
    if os.path.exists(checkpoint_path):
        print("🟢 Found a previous training checkpoint. Resuming training exactly where it left off...")
        # Load the last checkpoint
        model = YOLO(checkpoint_path)
        # Resume training
        results = model.train(resume=True)
    else:
        print("🔵 No checkpoint found. Starting new training from the beginning...")
        # Load the base model
        model = YOLO("yolov8n.pt")
        # Start training
        results = model.train(
            data="datasets/weapon_detection/data.yaml",  
            epochs=50,                        
            imgsz=640,                        
            batch=8,                          
            device="cpu",                     
            patience=10,                      
            project="runs/train",             
            name="guns_model",                
            verbose=True
        )

    print("\n✅ Training complete!")
    print("Best model saved at: runs/train/guns_model/weights/best.pt")
    print("\nNext step: update gui.py line 64 to use your new model:")
    print('  self.yolo_model = YOLO("runs/train/guns_model/weights/best.pt")')

if __name__ == "__main__":
    main()
