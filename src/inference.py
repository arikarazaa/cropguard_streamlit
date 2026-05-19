"""
inference.py — Single-image prediction API used by app.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import CLASS_NAMES, get_risk, get_val_transforms
from src.gradcam import GradCAM, get_target_layer
from src.model import CropGuardModel, load_model

IMG_SIZE = 224

class CropGuardPredictor:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_val_transforms(IMG_SIZE)
        self.demo_mode = (checkpoint_path is None) or (not os.path.exists(str(checkpoint_path)))
        if self.demo_mode:
            print("No checkpoint found — running in DEMO mode.")
            self.model = CropGuardModel()
            self.model.eval()
        else:
            self.model = load_model(checkpoint_path, self.device)
        self.model.to(self.device)
        target_layer = get_target_layer(self.model)
        self.gradcam  = GradCAM(self.model, target_layer)

    def predict(self, pil_image: Image.Image, top_k: int = 5) -> dict:
        original = np.array(pil_image.convert("RGB"))
        resized  = __import__("cv2").resize(original, (IMG_SIZE, IMG_SIZE))
        tensor   = self.transform(image=resized)["image"].unsqueeze(0).to(self.device)

        heatmap, class_idx, probs = self.gradcam(tensor)

        disease  = CLASS_NAMES[class_idx]
        conf     = float(probs[class_idx])
        risk     = get_risk(disease)

        top_idx  = probs.argsort()[::-1][:top_k]
        top_k_predictions = [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]

        overlay_img = GradCAM.overlay(resized, heatmap, alpha=0.45)
        gradcam_pil = GradCAM.to_pil(overlay_img)

        return {
            "disease":           disease,
            "confidence":        conf,
            "risk":              risk,
            "top_k":             top_k_predictions,
            "gradcam_pil":       gradcam_pil,
            "heatmap":           heatmap,
            "demo_mode":         self.demo_mode,
        }
