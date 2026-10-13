from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.aggregation import aggregate_eye_features
LABELS=["irf","hrf","pavf","favf","drt_me","vitreous_debris"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",default=ROOT/"data/processed/annotations.csv");p.add_argument("--output",default=ROOT/"data/processed/eye_features_annotations.csv");p.add_argument("--mode",choices=["annotation","prediction"],default="annotation");p.add_argument("--model-version");a=p.parse_args()
 f=aggregate_eye_features(pd.read_csv(a.input),LABELS,a.mode,a.model_version);f.to_csv(a.output,index=False);print(f"Wrote {len(f)} eye-level rows to {a.output}")
if __name__=="__main__":main()
