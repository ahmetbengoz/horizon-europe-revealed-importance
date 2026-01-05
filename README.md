# Learning Context-Dependent Criteria Importance from Revealed Policy Outcomes

This repository provides the full reproducibility pipeline for the paper:

Ahmet Bengoz,
"Learning Context-Dependent Criteria Importance from Revealed Policy Outcomes:
A Data-Driven Framework for Multi-Criteria Decision Making"

## Data
Raw project data are obtained from the CORDIS Horizon Europe open dataset:
https://data.europa.eu/data/datasets/cordis-eu-research-projects-under-horizon-europe-2021-2027

The file `cordis_projects_2021_2024.csv` contains 19,031 projects.

## Reproducibility
To reproduce all results:

1. Run `01_data_cleaning.py`
2. Run `02_proxy_construction.py`
3. Run `03_learning_model.py`
4. Run `04_cross_validation.py`
5. Run `05_statistical_tests.py`

All tables and figures reported in the paper and supplementary materials
are generated automatically.

## Environment
Python 3.12 (64-bit)

Required packages are listed in `requirements.txt`.
