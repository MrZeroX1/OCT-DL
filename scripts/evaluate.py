"""Evaluate a completed checkpoint; threshold optimization is exclusively validation-driven."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset import OCTBiomarkerDataset
from src.model import build_model
from src.metrics import binary_metrics,choose_threshold
def predict(model,loader):
 out=[];model.eval()
 with torch.no_grad():
  for b in loader:
   p=torch.sigmoid(model(b["image"])).numpy();out.extend(zip(b["patient_id"],b["target"].numpy(),b["mask"].numpy(),p))
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--val-csv",default=ROOT/"data/splits/val.csv");p.add_argument("--test-csv",default=ROOT/"data/splits/test.csv");p.add_argument("--output",default=ROOT/"reports/evaluation.json");a=p.parse_args();c=torch.load(a.checkpoint,map_location="cpu",weights_only=False);cfg=c["config"];names=cfg["model"]["biomarkers"];m=build_model(cfg["model"]["architecture"],len(names),False,cfg["model"]["dropout"]);m.load_state_dict(c["model_state"])
 def load(x):return predict(m,DataLoader(OCTBiomarkerDataset(x,names,cfg["data"]["image_size"]),batch_size=32))
 va,te=load(a.val_csv),load(a.test_csv);thresholds={};result={"threshold_source":"validation only","biomarkers":{}}
 for i,n in enumerate(names):
  vy,vp=zip(*[(x[1][i],x[3][i]) for x in va if x[2][i]]);ty,tp=zip(*[(x[1][i],x[3][i]) for x in te if x[2][i]]);thresholds[n]=choose_threshold(vy,vp);result["biomarkers"][n]=binary_metrics(ty,tp,thresholds[n])
 result["thresholds"]=thresholds;Path(a.output).write_text(json.dumps(result,indent=2));print(a.output)
if __name__=="__main__":main()
