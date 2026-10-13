from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.preprocessing import load_oct_image
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",default=ROOT/"data/processed/manifest.csv"); a=p.parse_args(); frame=pd.read_csv(a.manifest); failed=[]
 for r in frame.itertuples():
  try: load_oct_image(r.image_path)
  except Exception as e: failed.append({"image_id":r.image_id,"reason":str(e)})
 pd.DataFrame(failed).to_csv(ROOT/"data/processed/validation_failures.csv",index=False)
 if failed: raise SystemExit(f"{len(failed)} unreadable processed images")
 print(f"Validated {len(frame)} images; no unreadable paths.")
if __name__=="__main__": main()
