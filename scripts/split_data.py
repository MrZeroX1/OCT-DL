from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
LABELS=["irf","hrf","pavf","favf","drt_me","vitreous_debris"]
def assign(frame, seed):
 groups=frame.patient_id.astype(str).to_numpy(); n=len(frame); best=None
 # Repeated group draws select the candidate closest to requested volume and label prevalence.
 for attempt in range(200):
  outer=GroupShuffleSplit(1,test_size=.30,random_state=seed+attempt); train_idx,hold_idx=next(outer.split(frame,groups=groups))
  hold=frame.iloc[hold_idx]; splitter=GroupShuffleSplit(1,test_size=.5,random_state=seed+1000+attempt); val_rel,test_rel=next(splitter.split(hold,groups=hold.patient_id.astype(str)))
  parts=[frame.iloc[train_idx],hold.iloc[val_rel],hold.iloc[test_rel]]; sizes=np.array([len(x)/n for x in parts]); loss=np.abs(sizes-[.7,.15,.15]).sum()
  all_prev=frame[LABELS].mean(skipna=True)
  for x in parts: loss += (x[LABELS].mean(skipna=True).sub(all_prev).abs().fillna(.5).sum())*.2
  if best is None or loss<best[0]: best=(loss,parts)
 return best[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",default=ROOT/"data/processed/manifest.csv");p.add_argument("--seed",type=int,default=2026);a=p.parse_args()
 frame=pd.read_csv(a.manifest)
 # Preserve these rows in the manifest for provenance, but omit scans with no
 # supervised target from loaders: a fully unknown target cannot contribute a loss.
 frame=frame.dropna(subset=["patient_id","eye_id"]).dropna(how="all",subset=LABELS)
 if frame.patient_id.nunique()<7: raise SystemExit("Fewer than 7 independent patients: defensible 70/15/15 patient split is impossible.")
 train,val,test=assign(frame,a.seed); out=ROOT/"data/splits";out.mkdir(exist_ok=True)
 for name,x in zip(["train","val","test"],[train,val,test]): x.to_csv(out/f"{name}.csv",index=False)
 sets=[set(x.patient_id.astype(str)) for x in (train,val,test)]; overlap=any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)); dup=bool((set(train.image_id)&set(val.image_id)) or (set(train.image_id)&set(test.image_id)) or (set(val.image_id)&set(test.image_id)))
 lines=["# Split validation","",f"- Seed: {a.seed}","- Eligibility: images must retain patient/eye identifiers and at least one direct biomarker label.",f"- Patient overlap: {overlap}",f"- Image overlap: {dup}","", "| split | images | patients | eyes |"]
 lines += ["|---|---:|---:|---:|",*[f"| {n} | {len(x)} | {x.patient_id.nunique()} | {x.eye_id.nunique()} |" for n,x in zip(["train","val","test"],[train,val,test])]]
 for name,x in zip(["train","val","test"],[train,val,test]):
  lines.extend(["",f"## {name} label prevalence","",x[LABELS].agg(["count","sum","mean"]).round(4).to_markdown()])
 (ROOT/"reports/split_validation.md").write_text("\n".join(lines)+"\n",encoding="utf8")
 print("\n".join(lines[:8]))
if __name__=="__main__":main()
