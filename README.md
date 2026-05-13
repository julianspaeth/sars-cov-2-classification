# SARS-CoV-2 Drug Effect Classification

Machine learning models that integrate human protein-protein interaction networks with drug compound structures to predict the efficacy and cytotoxicity of chemical compounds against SARS-CoV-2.

## What this is

This repository contains the code for the analysis published as:

> Späth J, Wang RS, Humphrey M, Baumbach J, Loscalzo J (2024). **Machine learning-based integration of network features and chemical structure of compounds for SARS-CoV-2 drug effect analysis**. *CPT: Pharmacometrics & Systems Pharmacology*, 13(2), 257-269.
> [Link to paper](https://doi.org/10.1002/psp4.13076)

The work was conducted during a research stay at Joseph Loscalzo's lab at Harvard Medical School / Brigham and Women's Hospital.

## Approach

- **Graph-based features** from the human protein-protein interactome
- **Molecular structure features** from drug compounds
- **Integrated ML models** to predict drug efficacy and cytotoxicity against SARS-CoV-2
- **SHAP** for interpretation of feature contributions and biological plausibility

## Tech Stack

- Python
- scikit-learn
- SHAP
- pandas, NumPy
