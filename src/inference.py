from __future__ import annotations
from pathlib import Path
import json, yaml, torch
from .model import build_model
from .preprocessing import tensor_from_path, load_oct_image

def predict_biomarkers(image_path, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    image = load_oct_image(image_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]; model_cfg = config["model"]; names = model_cfg["biomarkers"]
    model = build_model(model_cfg["architecture"], len(names), False, model_cfg["dropout"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    with torch.no_grad(): probs = torch.sigmoid(model(tensor_from_path(image_path, config["data"]["image_size"]).unsqueeze(0)))[0].tolist()
    thresholds = ckpt.get("thresholds", {name: .5 for name in names})
    findings = {name: {"probability": float(p), "positive": bool(p >= thresholds.get(name,.5)), "threshold": float(thresholds.get(name,.5))} for name,p in zip(names,probs)}
    return {"image_path": str(Path(image_path)), "image_size": [image.width,image.height], "model_version": ckpt.get("model_version","unknown"), "findings": findings}

def prediction_json(image_path, checkpoint_path): return json.dumps(predict_biomarkers(image_path, checkpoint_path), indent=2)
