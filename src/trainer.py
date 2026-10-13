from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, torch
from torch.amp import GradScaler, autocast
from .metrics import binary_metrics

def set_seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def class_pos_weight(csv, labels):
 import pandas as pd
 f=pd.read_csv(csv); values=[]
 for name in labels:
  obs=pd.to_numeric(f[name],errors="coerce").dropna(); pos=obs.sum(); values.append(float((len(obs)-pos)/pos) if pos else 1.)
 return torch.tensor(values,dtype=torch.float32)

class Trainer:
 def __init__(self,model,criterion,optimizer,scheduler,device,config):
  self.model,self.criterion,self.optimizer,self.scheduler,self.device,self.config=model,criterion,optimizer,scheduler,device,config
  self.scaler=GradScaler("cuda",enabled=config["training"]["mixed_precision"] and device.type=="cuda")
 def _run(self,loader,training):
  self.model.train(training);total=0.;observed=0
  for batch in loader:
   x=batch["image"].to(self.device);y=batch["target"].to(self.device);m=batch["mask"].to(self.device)
   if training:self.optimizer.zero_grad(set_to_none=True)
   with autocast("cuda",enabled=self.scaler.is_enabled()): loss=self.criterion(self.model(x),y,m)
   if training:
    self.scaler.scale(loss).backward();self.scaler.unscale_(self.optimizer);torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config["training"]["gradient_clip_norm"]);self.scaler.step(self.optimizer);self.scaler.update()
   total+=loss.item()*m.sum().item();observed+=m.sum().item()
  return total/max(observed,1)
 def fit(self,train_loader,val_loader):
  cfg=self.config; directory=Path(cfg["training"]["checkpoint_dir"]);directory.mkdir(parents=True,exist_ok=True);best=float("inf");stale=0;history=[]
  for epoch in range(1,cfg["training"]["epochs"]+1):
   train_loss=self._run(train_loader,True)
   with torch.no_grad(): val_loss=self._run(val_loader,False)
   self.scheduler.step(val_loss); history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss})
   if val_loss<best:
    best=val_loss;stale=0;torch.save({"model_state":self.model.state_dict(),"config":cfg,"epoch":epoch,"best_val_loss":best,"model_version":"RetiSight-OCT"},directory/"best.pt")
   else: stale+=1
   (directory/"history.json").write_text(json.dumps(history,indent=2))
   print(f"epoch={epoch} train={train_loss:.5f} val={val_loss:.5f}")
   if stale>=cfg["training"]["early_stopping_patience"]:break
  return history
