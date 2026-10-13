# RetiSight retinal-feature handoff

`data/processed/eye_features_annotations.csv` is annotation-derived, not model output. Its `feature_source` is always `annotation`; future predicted exports must use `prediction` and record `model_version`.

Join on `patient_id`, `eye_id`, and `visit_id` when a visit is available. The curated Hugging Face subset does not expose an original visit or OCT-volume identifier, so those values remain missing rather than reconstructed. Keep missing feature values as missing: `*_missing` gives the number of unobserved scans, and a null aggregate means no observed scan.

For every biomarker, `*_mean`, `*_max`, and `*_positive_scan_proportion` summarize valid scans within the patient-eye-visit group; `n_scans` is the group denominator. Annotation features reflect the clinical labels; prediction features will instead reflect scan-level predicted probability.

Avoid leakage by splitting treatment-response cohorts by patient before fitting or threshold selection, retaining both eyes and all visits together. Do not mix annotation-derived features with model predictions without an explicit source indicator, and do not let post-treatment visits enter a baseline-response feature set. Suggested model inputs are the continuous mean/max features plus missingness indicators, with clinical timing aligned before treatment-response outcomes.
