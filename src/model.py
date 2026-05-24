"""
model.py — EfficientNet-B3 classifier with focal loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.dataset import NUM_CLASSES

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,label_smoothing=0.1):
        super().__init__(); self.gamma=gamma; self.ls=label_smoothing
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,label_smoothing=self.ls,reduction="none")
        pt=torch.exp(-ce)
        return (((1-pt)**self.gamma)*ce).mean()

class CropGuardModel(nn.Module):
    def __init__(self,model_name="efficientnet_b3",num_classes=NUM_CLASSES,dropout=0.3):
        super().__init__()
        self.backbone=timm.create_model(model_name,pretrained=True,num_classes=0,drop_rate=dropout)
        feat=self.backbone.num_features
        self.classifier=nn.Sequential(
            nn.Linear(feat,512),nn.BatchNorm1d(512),nn.SiLU(),
            nn.Dropout(dropout),nn.Linear(512,num_classes)
        )
    def forward(self,x):
        return self.classifier(self.backbone(x))

def load_model(checkpoint_path: str, device: str = "cpu") -> CropGuardModel:
    model = CropGuardModel()
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Remap keys: notebook used bb/head, app expects backbone/classifier
    remapped = {}
    for k, v in state.items():
        if k.startswith("bb."):
            remapped[k.replace("bb.", "backbone.", 1)] = v
        elif k.startswith("head."):
            remapped[k.replace("head.", "classifier.", 1)] = v
        else:
            remapped[k] = v

    # Handle any class count mismatch — load only matching keys
    model_state = model.state_dict()
    compatible  = {k: v for k, v in remapped.items()
                   if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    model.eval()
    return model.to(device)
