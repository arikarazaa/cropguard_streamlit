"""
gradcam.py — Grad-CAM with forward/backward hooks.
Reference: Selvaraju et al. ICCV 2017.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model; self._g=None; self._a=None
        target_layer.register_forward_hook(lambda m,i,o: setattr(self,"_a",o.detach()))
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self,"_g",go[0].detach()))

    def __call__(self,x,class_idx=None):
        self.model.eval()
        x=x.requires_grad_(True)
        logits=self.model(x)
        probs=F.softmax(logits,dim=1).squeeze().detach().cpu().numpy()
        if class_idx is None: class_idx=int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0,class_idx].backward()
        w=self._g.mean(dim=[2,3],keepdim=True)
        cam=F.relu((w*self._a).sum(dim=1,keepdim=True))
        cam=F.interpolate(cam,x.shape[2:],mode="bilinear",align_corners=False)
        cam=cam.squeeze().cpu().numpy()
        if cam.max()-cam.min()>1e-8:
            cam=(cam-cam.min())/(cam.max()-cam.min())
        return cam,class_idx,probs

    @staticmethod
    def overlay(original_rgb,heatmap,alpha=0.45):
        h,w=original_rgb.shape[:2]
        hm_resized=cv2.resize(heatmap,(w,h))
        hm_col=cv2.applyColorMap((hm_resized*255).astype(np.uint8),cv2.COLORMAP_JET)
        hm_col=cv2.cvtColor(hm_col,cv2.COLOR_BGR2RGB)
        return (alpha*hm_col+(1-alpha)*original_rgb).astype(np.uint8)

    @staticmethod
    def to_pil(arr): return Image.fromarray(arr.astype(np.uint8))

def get_target_layer(model):
    if hasattr(model.backbone,"blocks"): return model.backbone.blocks[-1]
    if hasattr(model.backbone,"layer4"):  return model.backbone.layer4[-1]
    raise ValueError("Cannot auto-detect target layer")
