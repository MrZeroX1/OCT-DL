from __future__ import annotations
import pandas as pd

def aggregate_eye_features(frame: pd.DataFrame, biomarkers: list[str], mode: str, model_version: str | None = None) -> pd.DataFrame:
    """Aggregate only observed annotations/probabilities, preserving their provenance."""
    required = {"patient_id", "eye_id", *biomarkers}
    if missing := required - set(frame.columns): raise ValueError(f"Missing columns: {sorted(missing)}")
    keys = [k for k in ["patient_id", "eye_id", "visit_id", "volume_id"] if k in frame]
    rows=[]
    for ids, group in frame.groupby(keys, dropna=False):
        ids = ids if isinstance(ids, tuple) else (ids,)
        row = dict(zip(keys, ids)); row.update({"feature_source": mode, "aggregation_method": "mean,max,positive_scan_proportion", "n_scans": len(group), "model_version": model_version})
        for b in biomarkers:
            observed = pd.to_numeric(group[b], errors="coerce").dropna()
            row[f"{b}_missing"] = int(len(group)-len(observed))
            row[f"{b}_mean"] = observed.mean() if len(observed) else None
            row[f"{b}_max"] = observed.max() if len(observed) else None
            row[f"{b}_positive_scan_proportion"] = (observed >= .5).mean() if len(observed) else None
        rows.append(row)
    return pd.DataFrame(rows)
