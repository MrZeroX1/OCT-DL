# Biomarker feasibility

Analysis uses only images with preserved patient and eye identifiers, because split safety is mandatory.

| biomarker | source | positives | negatives | missing | patients | suitable |
|---|---|---:|---:|---:|---:|---|
| irf | B4 | 2836 | 6572 | 60594 | 13 | yes |
| hrf | B1 | 6341 | 3067 | 60594 | 13 | yes |
| pavf | B2 | 2984 | 6424 | 60594 | 13 | yes |
| favf | B3 | 5222 | 4186 | 60594 | 13 | yes |
| drt_me | B5 | 3003 | 6405 | 60594 | 13 | yes |
| vitreous_debris | B6 | 4088 | 5320 | 60594 | 13 | yes |

SRF, DRIL, and EZ disruption are excluded from this model because they are not direct labels in the downloaded biomarker-detection image subset. They are not substituted with another biomarker.
