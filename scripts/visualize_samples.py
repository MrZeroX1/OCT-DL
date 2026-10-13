from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_oct_image, build_transform
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",default=ROOT/"data/processed/manifest.csv");p.add_argument("--output",default=ROOT/"reports/preprocessing_samples.png");p.add_argument("--n",type=int,default=4);a=p.parse_args(); f=pd.read_csv(a.manifest).head(a.n)
 fig,axes=plt.subplots(len(f),2,figsize=(10,3*len(f))); axes=axes if len(f)>1 else [axes]
 for ax,row in zip(axes,f.itertuples()):
  im=load_oct_image(row.image_path); ax[0].imshow(im,cmap="gray");ax[0].set_title(f"original {im.size}");ax[0].axis("off")
  t=build_transform(224)(im); display=(t*torch.tensor([.229,.224,.225]).view(3,1,1)+torch.tensor([.485,.456,.406]).view(3,1,1)).clamp(0,1);ax[1].imshow(display.permute(1,2,0));ax[1].set_title("224px normalized (display inverse-normalized)");ax[1].axis("off")
 fig.tight_layout();Path(a.output).parent.mkdir(exist_ok=True);fig.savefig(a.output,dpi=150);print(a.output)
if __name__=="__main__":main()
