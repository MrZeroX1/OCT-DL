# Split validation

- Seed: 2026
- Eligibility: images must retain patient/eye identifiers and at least one direct biomarker label.
- Patient overlap: False
- Image overlap: False

| split | images | patients | eyes |
|---|---:|---:|---:|
| train | 6762 | 9 | 9 |
| val | 1225 | 2 | 2 |
| test | 1421 | 2 | 2 |

## train label prevalence

|       |       irf |       hrf |      pavf |     favf |    drt_me |   vitreous_debris |
|:------|----------:|----------:|----------:|---------:|----------:|------------------:|
| count | 6762      | 6762      | 6762      | 6762     | 6762      |         6762      |
| sum   | 2131      | 4584      | 2290      | 3658     | 2129      |         2875      |
| mean  |    0.3151 |    0.6779 |    0.3387 |    0.541 |    0.3148 |            0.4252 |

## val label prevalence

|       |       irf |       hrf |      pavf |     favf |    drt_me |   vitreous_debris |
|:------|----------:|----------:|----------:|---------:|----------:|------------------:|
| count | 1225      | 1225      | 1225      | 1225     | 1225      |         1225      |
| sum   |  261      |  826      |  326      |  708     |  406      |          537      |
| mean  |    0.2131 |    0.6743 |    0.2661 |    0.578 |    0.3314 |            0.4384 |

## test label prevalence

|       |       irf |       hrf |     pavf |      favf |    drt_me |   vitreous_debris |
|:------|----------:|----------:|---------:|----------:|----------:|------------------:|
| count | 1421      | 1421      | 1421     | 1421      | 1421      |         1421      |
| sum   |  444      |  931      |  368     |  856      |  468      |          676      |
| mean  |    0.3125 |    0.6552 |    0.259 |    0.6024 |    0.3293 |            0.4757 |
