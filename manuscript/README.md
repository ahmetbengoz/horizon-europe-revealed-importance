# Manuscript support files

This folder documents how the manuscript outputs are linked to the reproducibility pipeline for the paper:

**Learning Outcome-Conditioned Proxy-Importance Profiles from Policy Data: A Decision-Support Framework for Horizon Europe**

## Purpose

The repository supports the empirical results reported in the manuscript, including:
- data audit
- proxy construction
- diagnostics
- transformed profile generation
- model comparison
- ablation and feature importance
- effect-size analysis
- downstream ranking demonstration

## Main tables and their output sources

### Table 2
**Distributional characteristics of raw proxy indicators**  
Source:
- `outputs/tables/Table_proxy_distribution.xlsx`

### Table 3
**Comparative predictive performance across models**  
Source:
- `outputs/tables/Table_model_comparison.xlsx`

### Table 4
**Late-period temporal validation results**  
Source:
- `outputs/tables/Table_temporal_holdout.xlsx`

### Table 5
**Funding-scheme differences in learned proxy-profile components**  
Source:
- `outputs/tables/Table_kruskal_effect_size.xlsx`

### Table 6
**Ablation results for contextual predictors**  
Source:
- `outputs/tables/Table_ablation.xlsx`

### Table 7
**Ranking comparison under alternative weighting schemes**  
Source:
- `outputs/tables/Table_ranking_comparison.xlsx`

### Table 8
**Weight scenarios used in the downstream demonstration**  
Source:
- `outputs/tables/Table_ranking_comparison.xlsx`

## Main figures and their output sources

### Figure 1
**Research workflow and empirical pipeline**  
Prepared for the manuscript as the conceptual workflow figure.

### Figure 2
**Spearman correlation matrix of raw proxies**  
Source:
- `outputs/figures/Figure_proxy_correlation.png`

### Figure 3
**Aggregated feature importance of contextual predictors**  
Source:
- `outputs/figures/Figure_feature_importance.png`

### Figure 4
**Ranking shifts under equal, learned, and entropy weighting**  
Source:
- `outputs/figures/Figure_rank_shift.png`

## Supplementary material candidates

The following files can be used as supplementary outputs:
- `outputs/tables/Table_normalization_robustness.xlsx`
- `outputs/tables/Table_proxy_transform_revision.xlsx`
- detailed ranking outputs contained in `Table_ranking_comparison.xlsx`

## Interpretation note

The manuscript should interpret the learned outputs as **outcome-conditioned proxy-importance profiles** derived from public funded-project data.

The manuscript should not describe these outputs as official evaluator weights.
