import os
import sys
import torch
import numpy as np
import cv2
from collections import deque

# Add the jepa directory to sys.path
jepa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "jepa"))
if jepa_path not in sys.path:
    sys.path.insert(0, jepa_path)

try:
    # Namespace packages in Python 3.3+ allow this without __init__.py
    from src.models.vision_transformer import vit_tiny, vit_small
    print("[SUCCESS] V-JEPA modules loaded successfully.")
except Exception as e:
    # If that fails, try importing directly if pathing is shifted
    try:
        from models.vision_transformer import vit_tiny, vit_small
        print("[SUCCESS] V-JEPA modules loaded successfully (Direct).")
    except Exception as e2:
        print(f"[ERROR] Error loading V-JEPA: {e}")
        print(f"[ERROR] Direct fallback also failed: {e2}")

class VJEPAEngine:
    def __init__(self):
        print("Initializing V-JEPA Anomaly Engine (Temporal Stream)...")
        # Use vit_small (384-dim) for a good balance of speed on CPU and accuracy
        # Note: We use random initialization because we are doing 'relative' anomaly detection
        # (Comparing current motion to the baseline of the last 10 seconds)
        self.device = "cpu"
        self.model = vit_small(img_size=224, num_frames=16, patch_size=16)
        self.model.eval()
        self.model.to(self.device)

        # Buffer for 16 frames (V-JEPA looks at short clips)
        self.frame_buffer = deque(maxlen=16)
        
        # Baseline memory: Stores embeddings of the last ~20 clips to know what 'normal' looks like
        self.baseline_memory = deque(maxlen=20)
        self.last_anomaly_score = 0.0

    def preprocess_frame(self, frame):
        """Convert OpenCV frame to V-JEPA input format (3, 224, 224)"""
        resized = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB and Normalize
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Transpose to (C, H, W)
        return torch.from_numpy(img).permute(2, 0, 1)

    def add_frame(self, frame):
        """Add a frame to the temporal buffer"""
        self.frame_buffer.append(self.preprocess_frame(frame))

    def compute_anomaly(self):
        """
        Takes the current 16-frame clip, generates a V-JEPA embedding,
        and compares it to the baseline memory using Cosine Similarity.
        """
        if len(self.frame_buffer) < 16:
            return {"score": 0.0, "label": "Calibrating..."}

        # Create clip tensor: (1, 3, 16, 224, 224)
        clip = torch.stack(list(self.frame_buffer), dim=1).unsqueeze(0)
        
        with torch.no_grad():
            # Generate V-JEPA spatial-temporal embedding
            # Note: JEPA outputs tokens, we take the mean to get a global 'scene fingerprint'
            tokens = self.model(clip)
            current_embedding = tokens.mean(dim=1).cpu().numpy().flatten()

        # If we have no baseline, this IS the baseline
        if len(self.baseline_memory) == 0:
            self.baseline_memory.append(current_embedding)
            return {"score": 0.0, "label": "Normal"}

        # Compare to baseline (Average Cosine Distance)
        # Higher distance = More different from normal = High Anomaly
        distances = []
        for base in self.baseline_memory:
            # Cosine similarity
            dot = np.dot(current_embedding, base)
            norm = np.linalg.norm(current_embedding) * np.linalg.norm(base)
            similarity = dot / (norm + 1e-9)
            distances.append(1.0 - similarity) # Distance is inverse of similarity

        avg_dist = np.mean(distances)
        
        # Smooth the score
        self.last_anomaly_score = (self.last_anomaly_score * 0.7) + (avg_dist * 0.3)
        
        # Logic: Update baseline only if it's 'mostly' normal to avoid learning the anomaly
        if self.last_anomaly_score < 0.4:
            self.baseline_memory.append(current_embedding)

        # Classification
        label = "Normal"
        if self.last_anomaly_score > 0.3: label = "Unusual Motion"
        if self.last_anomaly_score > 0.5: label = "HIGH ANOMALY"

        return {
            "score": float(self.last_anomaly_score),
            "label": label
        }
