"""
dataset.py — PlantVillage dataset + risk scoring.
NOTE: albumentations >=1.4 — RandomResizedCrop takes tuple, CenterCrop takes ints.
"""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry___Powdery_mildew","Cherry___healthy",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot","Corn___Common_rust",
    "Corn___Northern_Leaf_Blight","Corn___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot",
    "Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot",
    "Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy",
]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

RISK_DB = {
    "Tomato___Late_blight":        ("Critical",62,"Apply copper-based fungicide immediately. Remove infected leaves.",24),
    "Tomato___Early_blight":       ("High",38,"Apply chlorothalonil; increase plant spacing.",48),
    "Tomato___Bacterial_spot":     ("High",35,"Apply copper bactericide; avoid overhead irrigation.",48),
    "Tomato___Septoria_leaf_spot": ("Moderate",28,"Remove infected leaves; apply protective fungicide.",72),
    "Tomato___Leaf_Mold":          ("Moderate",20,"Improve ventilation; apply fungicide.",72),
    "Tomato___Target_Spot":        ("Moderate",22,"Apply fungicide preventatively.",72),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":("Critical",80,"Remove infected plants; control whitefly vectors.",12),
    "Tomato___Tomato_mosaic_virus":("High",45,"Remove infected plants; disinfect all tools.",24),
    "Potato___Late_blight":        ("Critical",70,"Apply metalaxyl + mancozeb; destroy infected haulms.",12),
    "Potato___Early_blight":       ("High",40,"Apply chlorothalonil; ensure adequate potassium.",48),
    "Corn___Northern_Leaf_Blight": ("High",30,"Apply strobilurin fungicide.",48),
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot":("Moderate",24,"Apply triazole fungicide.",72),
    "Corn___Common_rust":          ("Low",15,"Monitor; apply fungicide if >50% leaf area affected.",96),
    "Apple___Apple_scab":          ("Moderate",24,"Apply captan or dodine; rake fallen leaves.",72),
    "Apple___Black_rot":           ("High",45,"Prune infected tissue; apply copper fungicide.",48),
    "Apple___Cedar_apple_rust":    ("Moderate",20,"Apply myclobutanil.",72),
    "Grape___Black_rot":           ("Critical",55,"Apply mancozeb; remove mummified berries.",24),
    "Grape___Esca_(Black_Measles)":("High",35,"Prune infected wood.",48),
}

def get_risk(class_name: str) -> dict:
    if "healthy" in class_name.lower():
        return {"level":"None","yield_loss_pct":0,"action":"Healthy crop — continue monitoring.","act_within_hours":None}
    level,loss,action,hours = RISK_DB.get(class_name,("Moderate",20,"Consult local agronomist.",72))
    return {"level":level,"yield_loss_pct":loss,"action":action,"act_within_hours":hours}

def get_val_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size+32, img_size+32),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

def get_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop((img_size,img_size),scale=(0.7,1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30,p=0.5),
        A.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,p=0.7),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

class PlantVillageDataset(Dataset):
    def __init__(self,data_dir,transform=None,max_per_class=None):
        self.transform=transform; self.samples=[]
        classes=sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
        self.classes=classes; self.class_to_idx={c:i for i,c in enumerate(classes)}
        for cls_dir in Path(data_dir).iterdir():
            if not cls_dir.is_dir(): continue
            imgs=[p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]
            if max_per_class: imgs=imgs[:max_per_class]
            label=self.class_to_idx.get(cls_dir.name)
            if label is None: continue
            self.samples.extend([(p,label) for p in imgs])
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        path,label=self.samples[idx]
        img=np.array(Image.open(path).convert("RGB"))
        if self.transform: img=self.transform(image=img)["image"]
        return img,label
