"""
train.py — Training entrypoint.
Run: python src/train.py --data_dir data/plantvillage --epochs 80
Best done on Colab (free GPU) — see notebooks/CropGuard_Training.ipynb
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score
from src.dataset import PlantVillageDataset, get_train_transforms, get_val_transforms, NUM_CLASSES
from src.model import CropGuardModel, FocalLoss

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="data/plantvillage")
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--img_size",type=int,default=224)
    p.add_argument("--output_dir",default="models")
    p.add_argument("--max_per_class",type=int,default=None)
    return p.parse_args()

def main():
    args=parse_args()
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    full_ds=PlantVillageDataset(args.data_dir,transform=get_train_transforms(args.img_size),max_per_class=args.max_per_class)
    n=len(full_ds); n_val=int(n*0.15); n_test=int(n*0.10); n_train=n-n_val-n_test
    train_ds,val_ds,test_ds=random_split(full_ds,[n_train,n_val,n_test],generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform=get_val_transforms(args.img_size)
    test_ds.dataset.transform=get_val_transforms(args.img_size)

    train_loader=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader  =DataLoader(val_ds,  args.batch_size,shuffle=False,num_workers=2,pin_memory=True)

    model=CropGuardModel().to(DEVICE)
    for p in model.backbone.parameters(): p.requires_grad=False
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    criterion=FocalLoss()
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    best_f1=0

    UNFREEZE=10
    for epoch in range(args.epochs):
        if epoch==UNFREEZE:
            print(f"\n[Epoch {epoch}] Unfreezing backbone...")
            for p in model.backbone.parameters(): p.requires_grad=True
            optimizer.add_param_group({"params":model.backbone.parameters(),"lr":args.lr*0.1})

        model.train(); train_loss=0
        for imgs,labels in tqdm(train_loader,desc=f"Ep {epoch+1}/{args.epochs} train",leave=False):
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad(); loss=criterion(model(imgs),labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); train_loss+=loss.item()

        model.eval(); preds_all=[]; targets_all=[]
        with torch.no_grad():
            for imgs,labels in tqdm(val_loader,desc=f"Ep {epoch+1}/{args.epochs} val",leave=False):
                logits=model(imgs.to(DEVICE))
                preds_all.extend(logits.argmax(1).cpu().numpy())
                targets_all.extend(labels.numpy())

        scheduler.step()
        f1=f1_score(targets_all,preds_all,average="macro",zero_division=0)
        acc=sum(p==t for p,t in zip(preds_all,targets_all))/len(targets_all)
        print(f"Ep {epoch+1:3d}  loss={train_loss/len(train_loader):.4f}  acc={acc:.4f}  F1={f1:.4f}")

        if f1>best_f1:
            best_f1=f1
            torch.save({"model_state_dict":model.state_dict(),"class_names":full_ds.classes,"val_f1":f1,"val_acc":acc},
                       f"{args.output_dir}/best_model.pth")
            print(f"  ⭐ Saved best model (F1={f1:.4f})")

    print(f"\nDone. Best F1: {best_f1:.4f}")

if __name__=="__main__": main()
