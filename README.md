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
