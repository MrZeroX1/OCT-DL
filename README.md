# RetiSight OCT biomarker detection

This project prepares genuine OLIVES OCT scans for multi-label classification and deliberately does **not** run optimization during setup.

## Environment

Python 3.10–3.14 is required; PyTorch currently publishes no wheel for Python 3.15. On this workstation the prepared environment is `.venv` using Python 3.13:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Preparation

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/inspect_data.py
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/validate_data.py
.\.venv\Scripts\python.exe scripts/split_data.py
.\.venv\Scripts\python.exe scripts/visualize_samples.py
.\.venv\Scripts\python.exe scripts/export_features.py --mode annotation
```

The default download contains all 32 public training and two test Parquet files from the `biomarker_detection` configuration, plus Zenodo's label package. This is about 15 GB and is necessary for a defensible patient-level split; use `--test-only` only for inspection (its identifiers are anonymized).

## Labels and provenance

The selected source is [gOLIVES/OLIVES_Dataset](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset), `biomarker_detection`, with its six direct labels. The Zenodo source is record 7105232 (CC-BY-4.0); the Hugging Face distribution declares MIT. `configs/biomarker_mapping.yaml` documents direct mappings and exclusions. The full Zenodo label CSV contains SRF, DRIL, and EZ annotations but its 33.9-GB image archive is intentionally not duplicated during the biomarker-subset setup.

## Train only when reviewed

```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/train.yaml
```

This is intentionally not executed by any preparation command. After training, run evaluation using a held-out patient set:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best.pt
```

For annotation features: `python scripts/export_features.py --mode annotation`. For prediction features, pass a scan-level prediction table and `--mode prediction --model-version <version>`.
