"""Extract actual OCT bytes from the OLIVES parquet and construct an auditable manifest."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.preprocessing import image_statistics
MAP={"B4":"irf","B1":"hrf","B2":"pavf","B3":"favf","B5":"drt_me","B6":"vitreous_debris"}
def image_bytes(cell):
 if isinstance(cell,dict): return cell.get("bytes")
 return getattr(cell,"get",lambda *_:None)("bytes")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--raw-dir",default=ROOT/"data/raw/hf_biomarker"); p.add_argument("--image-size",type=int,default=224); args=p.parse_args()
 raw=Path(args.raw_dir); files=sorted(raw.rglob("*.parquet"));
 if not files: raise SystemExit("No source parquet files. Run scripts/download_data.py first.")
 out=ROOT/"data/processed"; images=out/"images"; images.mkdir(parents=True,exist_ok=True); rejected=[]; rows=[]
 for parquet in files:
  frame=pd.read_parquet(parquet)
  for index, item in frame.iterrows():
   blob=image_bytes(item["Image"])
   if not blob: rejected.append({"source_file":str(parquet),"source_row":index,"reason":"missing_image_bytes"}); continue
   digest=hashlib.sha256(blob).hexdigest(); path=images/f"{digest}.png"
   try:
    with Image.open(BytesIO(blob)) as im: im.load(); rgb=im.convert("RGB"); stats=image_statistics(rgb); rgb.resize((args.image_size,args.image_size),Image.Resampling.LANCZOS).save(path,"PNG")
    if not stats["contrast_nonzero"]: raise ValueError("zero contrast")
   except Exception as exc: rejected.append({"source_file":str(parquet),"source_row":index,"reason":str(exc)}); path.unlink(missing_ok=True); continue
   labels={dst: pd.to_numeric(item.get(src),errors="coerce") for src,dst in MAP.items()}
   patient_id=item.get("Patient_ID"); eye_id=item.get("Eye_ID")
   rows.append({"image_id":digest,"image_path":str(path.resolve()),"source_file":parquet.name,"source_row":int(index),"source_image_name":item["Image"].get("path") if isinstance(item["Image"],dict) else None,"patient_id":patient_id,"eye_id":eye_id,"visit_id":np.nan,"volume_id":np.nan,"training_eligible":bool(pd.notna(patient_id) and pd.notna(eye_id)),**stats,**labels})
 manifest=pd.DataFrame(rows).drop_duplicates("image_id",keep="first"); manifest.to_csv(out/"manifest.csv",index=False); manifest.to_csv(out/"annotations.csv",index=False)
 pd.DataFrame(rejected).to_csv(out/"rejected_images.csv",index=False)
 stats={"source_parquet_files":[str(x) for x in files],"source_rows":int(sum(len(pd.read_parquet(x,columns=["Patient_ID"])) for x in files)),"usable_unique_images":len(manifest),"training_eligible_images":int(manifest.training_eligible.sum()),"anonymized_images_excluded_from_patient_split":int((~manifest.training_eligible).sum()),"duplicates_removed":len(rows)-len(manifest),"rejected_rows":len(rejected),"patients":int(manifest.patient_id.nunique()),"eyes":int(manifest.eye_id.nunique()),"image_size":args.image_size,"label_columns":list(MAP.values())}
 (ROOT/"reports/dataset_statistics.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
 (ROOT/"reports/dataset_audit.md").write_text("# Dataset audit\n\n"+"\n".join(f"- **{k}**: {v}" for k,v in stats.items())+"\n\nOriginal parquet bytes remain under `data/raw`; processed PNGs are derived copies. Missing source labels are retained as blank values, never converted to negatives.\n",encoding="utf8")
 eligible=manifest[manifest.training_eligible].copy(); distribution=[]
 for source,key in MAP.items():
  values=pd.to_numeric(eligible[key],errors="coerce"); observed=values.notna().sum(); distribution.append({"biomarker":key,"source_column":source,"positive":int((values==1).sum()),"negative":int((values==0).sum()),"missing":int(values.isna().sum()),"patients":int(eligible.loc[values.notna(),"patient_id"].nunique()),"positive_fraction_among_observed":float((values==1).sum()/observed) if observed else None})
 pd.DataFrame(distribution).to_csv(ROOT/"reports/biomarker_distribution.csv",index=False)
 feasibility=["# Biomarker feasibility", "", "Analysis uses only images with preserved patient and eye identifiers, because split safety is mandatory.", "", "| biomarker | source | positives | negatives | missing | patients | suitable |", "|---|---|---:|---:|---:|---:|---|"]
 feasibility += [f"| {x['biomarker']} | {x['source_column']} | {x['positive']} | {x['negative']} | {x['missing']} | {x['patients']} | {'yes' if x['positive'] > 0 and x['negative'] > 0 else 'no'} |" for x in distribution]
 feasibility += ["", "SRF, DRIL, and EZ disruption are excluded from this model because they are not direct labels in the downloaded biomarker-detection image subset. They are not substituted with another biomarker."]
 (ROOT/"reports/biomarker_feasibility.md").write_text("\n".join(feasibility)+"\n",encoding="utf8")
 print(json.dumps(stats,indent=2))
if __name__=="__main__": main()
