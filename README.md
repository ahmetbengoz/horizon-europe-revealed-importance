# Learning Outcome-Conditioned Proxy-Importance Profiles from Policy Data: A Decision-Support Framework for Horizon Europe

This repository provides the reproducibility pipeline for the paper:

**Ahmet Bengoz, "Learning Outcome-Conditioned Proxy-Importance Profiles from Policy Data: A Decision-Support Framework for Horizon Europe"**

## Overview

The study develops a reproducible decision-support framework for learning **outcome-conditioned proxy-importance profiles** from public Horizon Europe funded-project data.

The repository implements the full empirical workflow:
- data audit
- proxy construction
- diagnostics
- transformed profile generation
- model comparison
- ablation and effect-size analysis
- downstream ranking demonstration

## Important interpretation note

This repository implements **proxy-profile inference** from public funded-project metadata.

It does **not** claim to recover:
- official evaluator weights
- full latent preference structures
- confidential evaluation rules used by the funding authority

The outputs should be interpreted as **policy-outcome-conditioned proxy-importance profiles**.

## Data source

The empirical analysis is based on the public CORDIS dataset for Horizon Europe projects (2021–2027), available through the European Union open data portal.

Raw dataset file expected by the scripts:

`data/cordis_he_projects.csv`

## Empirical sample used in the study

- Raw audited dataset: **18,374** project records
- Valid profile-analysis sample: **17,917** project records

## Repository structure

```text
horizon-europe-revealed-importance/
│
├── data/
│   └── README.md
│
├── manuscript/
│   └── README.md
│
├── outputs/
│   ├── tables/
│   └── figures/
│
├── scripts/
│   ├── 01_data_audit.py
│   ├── 02_proxy_construction.py
│   ├── 03_proxy_diagnostics.py
│   ├── 03b_proxy_transform_revision.py
│   ├── 04_model_comparison.py
│   ├── 05_ablation_and_importance_fast.py
│   ├── 05b_effect_size_kruskal.py
│   └── 06_downstream_demo.py
│
├── requirements.txt
├── LICENSE
└── README.md
## Reproducibility pipeline
Run the scripts in the following order:

01_data_audit.py
Audits the raw Horizon Europe dataset and checks required fields.

02_proxy_construction.py
Constructs the raw proxy variables and the initial project-level profile dataset.

03_proxy_diagnostics.py
Produces proxy diagnostics and normalization sensitivity checks.

03b_proxy_transform_revision.py
Applies the revised transformed profile-generation procedure used in the final study.

04_model_comparison.py
Compares predictive models using error-based and rank-based metrics.

05_ablation_and_importance_fast.py
Performs contextual ablation analysis and feature-importance aggregation.

05b_effect_size_kruskal.py
Computes Kruskal–Wallis statistics and epsilon-squared effect sizes across funding schemes.

06_downstream_demo.py
Produces the downstream ranking demonstration under equal, learned, and entropy weighting scenarios.

## Main outputs
# Tables
outputs/tables/Table_proxy_distribution.xlsx
outputs/tables/Table_model_comparison.xlsx
outputs/tables/Table_temporal_holdout.xlsx
outputs/tables/Table_ablation.xlsx
outputs/tables/Table_feature_importance.xlsx
outputs/tables/Table_kruskal_effect_size.xlsx
outputs/tables/Table_ranking_comparison.xlsx
outputs/tables/Table_normalization_robustness.xlsx
outputs/tables/Table_proxy_transform_revision.xlsx

# Figures
outputs/figures/Figure_proxy_correlation.png
outputs/figures/Figure_feature_importance.png
outputs/figures/Figure_rank_shift.png

## Main methodological components
### The final empirical design includes:
### proxy diagnostics
### revised transformation and percentile-based scaling
### cross-validated model comparison
### late-period temporal validation
### ablation analysis
### feature-importance analysis
### effect-size analysis across funding schemes
### downstream ranking comparison

## Software requirements

The required Python packages are listed in:

requirements.txt

Typical dependencies include:

pandas
numpy
scikit-learn
scipy
matplotlib
openpyxl

## Data availability

The repository documents the reproducibility pipeline and output structure for the study.

The raw project data are obtained from the public CORDIS Horizon Europe dataset.
Users should download the raw file and place it in the data/ folder as:

cordis_he_projects.csv

## Citation note

If you use this repository, please cite the associated paper once published.
