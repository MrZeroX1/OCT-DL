"""Download public, non-duplicated OLIVES biomarker parquet shards and Zenodo labels."""
from __future__ import annotations
import argparse, json, shutil, sys, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from huggingface_hub import HfApi, hf_hub_download

REPO="gOLIVES/OLIVES_Dataset"; ZENODO="https://zenodo.org/api/records/7105232/files/OLIVES_Dataset_Labels.zip/content"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--max-shards",type=int,default=0,help="Number of train shards; 0 means all 32")
    p.add_argument("--test-only",action="store_true",help="Download only the anonymized test split (not usable for patient-level splitting)")
    args=p.parse_args(); raw=ROOT/"data/raw"; raw.mkdir(parents=True,exist_ok=True)
    labels=raw/"zenodo_labels/OLIVES_Dataset_Labels"
    if not labels.exists():
        archive=raw/"OLIVES_Dataset_Labels.zip"; print("Downloading Zenodo labels...")
        urllib.request.urlretrieve(ZENODO, archive); zipfile.ZipFile(archive).extractall(raw/"zenodo_labels"); archive.unlink(missing_ok=True)
    files=HfApi().list_repo_files(REPO, repo_type="dataset")
    test=sorted(x for x in files if x.startswith("biomarker_detection/test-") and x.endswith(".parquet"))
    train=sorted(x for x in files if x.startswith("biomarker_detection/train-") and x.endswith(".parquet"))
    selected=test if args.test_only else test + (train if args.max_shards==0 else train[:args.max_shards])
    target=raw/"hf_biomarker"; target.mkdir(exist_ok=True)
    downloaded=[]
    for name in selected:
        path=hf_hub_download(REPO, name, repo_type="dataset", local_dir=target)
        downloaded.append({"repo_file":name,"local_path":str(Path(path).resolve()),"bytes":Path(path).stat().st_size})
        print(f"Available: {name}")
    (raw/"download_manifest.json").write_text(json.dumps({"source":REPO,"configuration":"biomarker_detection","files":downloaded},indent=2),encoding="utf8")
    print(f"Downloaded {len(downloaded)} parquet shards; labels at {labels}")
if __name__=="__main__": main()
