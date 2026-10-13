"""Training entrypoint. Intentionally not invoked by project preparation."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset import OCTBiomarkerDataset
from src.model import MaskedBCEWithLogitsLoss,build_model
from src.trainer import Trainer,class_pos_weight,set_seed
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());set_seed(cfg["seed"])
 device=torch.device("cuda" if cfg["device"]=="auto" and torch.cuda.is_available() else cfg["device"] if cfg["device"]!="auto" else "cpu");names=cfg["model"]["biomarkers"];d=cfg["data"]
 train=OCTBiomarkerDataset(d["train_csv"],names,d["image_size"],True);val=OCTBiomarkerDataset(d["val_csv"],names,d["image_size"])
 kw={"batch_size":cfg["training"]["batch_size"],"num_workers":d["num_workers"],"pin_memory":device.type=="cuda","persistent_workers":d["persistent_workers"] and d["num_workers"]>0}
 model=build_model(cfg["model"]["architecture"],len(names),cfg["model"]["pretrained"],cfg["model"]["dropout"],cfg["model"]["freeze_backbone"]).to(device)
 loss=MaskedBCEWithLogitsLoss(class_pos_weight(d["train_csv"],names).to(device));opt=torch.optim.AdamW(filter(lambda x:x.requires_grad,model.parameters()),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"]);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=cfg["training"]["scheduler_factor"],patience=cfg["training"]["scheduler_patience"])
 if cfg["training"].get("resume"):
  state=torch.load(cfg["training"]["resume"],map_location=device,weights_only=False);model.load_state_dict(state["model_state"])
 Trainer(model,loss,opt,scheduler,device,cfg).fit(DataLoader(train,shuffle=True,**kw),DataLoader(val,shuffle=False,**kw))
if __name__=="__main__":main()
