# RetiSight OCT training readiness

## Status

**READY FOR TRAINING** for the six directly labelled OLIVES biomarkers, subject to the cohort limitations below. No model training, optimizer step, checkpoint, or model evaluation was run during preparation.

## Data provenance and preparation

- Images and six direct labels: Hugging Face [`gOLIVES/OLIVES_Dataset`](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset), `biomarker_detection` configuration, 32 training and 2 test Parquet shards.
- Supplementary original annotations: Zenodo record [7105232](https://zenodo.org/records/7105232), `OLIVES_Dataset_Labels.zip` (CC-BY-4.0). The Hugging Face distribution declares MIT.
- The upstream `olivesgatech/OLIVES_Dataset` source repository was cloned locally as a reference. Windows cannot check out several paths containing `:`; this does not affect the downloaded Parquet data or the pipeline.
- Downloaded rows: **82,693**. Unique readable OCT images: **74,412**. Rejected/corrupt images: **0**. Duplicate image bytes removed: **8,281**.
- Images with preserved patient and eye IDs: **70,002**, across **87 patients** and **96 eyes**. The remaining **4,410** images have anonymized IDs and are never used in the patient split.
- The source Parquet does not expose visit or OCT-volume IDs. `visit_id` and `volume_id` remain missing rather than being inferred.
- The supervised cohort consists of **9,408** scans with at least one direct biomarker label from **13** patients; all unlabelled scans are retained in the manifest but excluded from supervised loaders.

## Supported outputs

The EfficientNet-B0 head has six logits: `irf`, `hrf`, `pavf`, `favf`, `drt_me`, and `vitreous_debris`. Their direct source mappings and definitions are in `configs/biomarker_mapping.yaml`.

The requested **IRF** and **HRF** are directly supported (`B4` and `B1`). **SRF**, **DRIL**, and **EZ disruption** are excluded from the model: the full Zenodo label file contains them, but the downloaded biomarker image subset has no safe, direct image-level mapping to those columns. They were not inferred or recoded.

Observed-label distributions are in `reports/biomarker_distribution.csv`; positive counts range from 2,836 (IRF) to 6,341 (HRF) out of 9,408 observed labels. This class imbalance is handled with training-set `pos_weight` in masked BCE loss.

## Patient-level split

| Split | Scans | Patients | Eyes |
|---|---:|---:|---:|
| Train | 6,762 | 9 | 9 |
| Validation | 1,225 | 2 | 2 |
| Test | 1,421 | 2 | 2 |

The seeded split has no patient or image overlap. Details and per-label prevalence are in `reports/split_validation.md`. The independent patient count is small, so test confidence intervals will be wide; threshold optimization remains validation-only.

## Completed checks

- All 74,412 processed images were opened and verified; `validation_failures.csv` is empty.
- Real OCT image loading produced batches shaped `(2, 3, 224, 224)`.
- A pre-trained EfficientNet-B0 was loaded, produced `(2, 6)` logits, and a masked BCE loss was computed without an optimizer step.
- Unit tests: `2 passed`.
- `scripts/train.py --help`, `scripts/evaluate.py --help`, module compilation, dependency integrity, patient overlap checks, preprocessing visualization, and annotation-mode eye-level export all passed.
- `reports/preprocessing_samples.png` shows genuine OCT preprocessing samples; `data/processed/eye_features_annotations.csv` contains **98** annotation-derived eye-level rows.

## Configuration and commands

The configuration is `configs/train.yaml`: EfficientNet-B0 with pre-trained weights, 224px input, AdamW, masked `BCEWithLogitsLoss`, class weighting, AMP when CUDA is available, ReduceLROnPlateau, clipping, early stopping, checkpointing, and resume support.

The prepared local environment currently has a CPU PyTorch build and no detected NVIDIA utility. The code selects CUDA automatically when a CUDA-enabled PyTorch build and GPU are available; otherwise it safely uses CPU.

Begin training only when ready:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/train.yaml
```

After training:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best.pt
.\.venv\Scripts\python.exe scripts/export_features.py --mode annotation
```

For prediction-mode feature export, create a scan-level prediction table from a trained checkpoint and invoke `scripts/export_features.py --mode prediction --model-version <version>`.
