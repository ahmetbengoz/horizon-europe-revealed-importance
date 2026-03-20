# Data

This folder documents the raw data source used in the study:

**Learning Outcome-Conditioned Proxy-Importance Profiles from Policy Data: A Decision-Support Framework for Horizon Europe**

## Data source

The empirical analysis is based on the public CORDIS dataset for Horizon Europe projects (2021–2027), available through the European Union open data portal.

Dataset source:
CORDIS – EU research projects under Horizon Europe (2021–2027)

## Input file expected by the scripts

Place the raw dataset in this folder using the following file name:

`cordis_he_projects.csv`

## Scope of the data used in the study

The analysis uses publicly available funded-project records and project-level metadata, including:
- project identifiers
- dates
- EU contribution
- call descriptors
- funding scheme
- legal basis
- objective text
- keywords

The repository does **not** include private evaluation files, reviewer score sheets, rejected proposals, or official evaluator-specific weighting information.

## Important interpretation note

The study uses funded-project metadata to construct **proxy-importance profiles**.  
It does **not** claim to recover official evaluator weights or full latent preference structures.

## Data-processing note

The reproducibility pipeline begins with:
- data audit
- proxy construction
- diagnostics
- transformed profile generation

The final empirical workflow is implemented through the scripts in the `scripts/` folder.

## Practical note

If you download a fresh version of the CORDIS dataset, save it as:

`cordis_he_projects.csv`

and place it in this folder before running the scripts.
