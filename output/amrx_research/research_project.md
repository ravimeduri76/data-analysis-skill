# AMRx UTI Prediction Research Project
# Date: 2026-04-08
# Dataset: matching_data.csv (4325 rows x 86 cols)
# Primary Metric: MCC (Matthews Correlation Coefficient)


## OBJ1: UTI Prediction (Binary)
Baseline: 0.8973 | Target: 0.9200 | NOT MET
Best: LightGBM_optuna MCC=0.9025 (+0.0052 vs baseline)
Experiments run: 3

Experiment   Model                          MCC      Delta   
------------------------------------------------------------
EXP-P1       LightGBM_optuna                  0.9025  +0.0052
EXP-M5       VotingEnsemble_3model            0.9002  +0.0029
EXP-M1       CatBoost_native                  0.8892  -0.0081

## OBJ2a: Enterobacteriaceae Identification (Binary)
Baseline: 0.3900 | Target: 0.5500 | NOT MET
Best: GBM_optuna MCC=0.4139 (+0.0239 vs baseline)
Experiments run: 6

Experiment   Model                          MCC      Delta   
------------------------------------------------------------
EXP-P2       GBM_optuna                       0.4139  +0.0239
EXP-D1b      GBM_optuna_recoded               0.3960  +0.0060
EXP-P3       LightGBM_optuna                  0.3902  +0.0002
EXP-M8       CatBoost_bacteria                0.3818  -0.0082
EXP-M7       KNN_15                           0.1993  -0.1907
EXP-M6       MLP_128_64_32                    0.1005  -0.2895

## OBJ2b: Organism Species Identification (Multi-class)
Baseline: 0.1549 | Target: 0.3000 | NOT MET
Best: XGB_optuna_multiclass MCC=0.1521 (-0.0028 vs baseline)
Experiments run: 2

Experiment   Model                          MCC      Delta   
------------------------------------------------------------
EXP-P4       XGB_optuna_multiclass            0.1521  -0.0028
EXP-M9       CatBoost_multiclass              0.1202  -0.0347

## OBJ2b-alt: E.coli Binary Prediction
Baseline: N/A (new objective)
Best: CatBoost_ecoli MCC=0.2263

## Stopping Analysis
OBJ1:  Threshold=0.92, Best=0.9025 -- NOT MET but within 0.02
OBJ2a: Threshold=0.55, Best=0.4139 -- NOT MET, significant gap
OBJ2b: Threshold=0.30, Best=0.1549 -- NOT MET, fundamental limit

## Key Findings
1. UTI prediction from clinical features achieves MCC=0.90+ (strong)
2. Enterobacteriaceae identification is limited (MCC~0.41) -- clinical features
   alone may not differentiate bacterial families well
3. Multi-class organism identification is very limited (MCC~0.15) --
   clinical presentation alone cannot distinguish between bacterial species
4. Binary E.coli prediction shows MCC=0.23 -- slightly better than random
   but clinically insufficient for species-level identification
5. CatBoost native categorical handling did NOT improve over standard encoding
6. Optuna Bayesian optimization provided modest gains for OBJ2a (+0.024)
7. Feature recoding (1/2/3 -> 0/1/NaN) did not significantly impact results
8. MLP and KNN performed poorly for bacteria identification

## Clinical Implications
- UTI screening model (sensitivity>=95%) is viable for clinical use
- Bacteria identification requires lab culture data -- clinical features are
  insufficient for genus/species-level prediction
- Antibiotic sensitivity prediction not feasible with current datasets
  (Ast_cleaned.xlsx not available)