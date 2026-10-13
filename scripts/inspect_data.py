from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--raw-dir",default=ROOT/"data/raw/hf_biomarker"); args=p.parse_args()
 files=sorted(Path(args.raw_dir).rglob("*.parquet"));
 if not files: raise SystemExit("No parquet shards. Run scripts/download_data.py first.")
 for f in files:
  df=pd.read_parquet(f); print(json.dumps({"file":str(f),"rows":len(df),"columns":list(df.columns),"sample":df.drop(columns=["Image"],errors="ignore").head(2).to_dict("records")},indent=2,default=str))
if __name__=="__main__": main()
