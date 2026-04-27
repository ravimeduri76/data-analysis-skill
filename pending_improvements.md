
## 2026-03-25 09:11 UTC

- [ ] [REVIEW] Domain mismatch on 'C:\Users\Ravi\Downloads\cancer_variant_prioritization_dataset.csv': domain_mismatch: detected Healthcare, actual Clinical Genomics — needs Clinical Genomics keywords in domain_overrides.json (genomics, variant, mutation, allele, chromosome, CADD, gnomAD, pathogenicity)

## 2026-04-17 04:35 UTC — FAILURE ANALYSIS

- Dataset: `tiagoadrianunes/imdb-top-5000-tv-shows` (IMDb Top 5000 TV Shows)
- Detected domain: Education
- Shape: 2631 rows x 12 cols
- Chosen target: `startYear` (task=regression)
- Failure mode: **low_signal**
- Quality gate: successful_trials=8 (need >=3), best_R2=0.0295 (need >0.05)
- Root cause: bad target choice — startYear is not predictable from other features; domain misdetected as Education
- [ ] Improve target selection heuristic to deprioritise year/date-like columns for regression

## 2026-04-18 09:49 UTC — FAILURE ANALYSIS

- Dataset: `mdmahfuzsumon/pharma-dataset-drug-classes-interactions-and-cli-pr` (Pharma Dataset-Drug Classes, Interactions & Cli Pr)
- Detected domain: Pharmaceutical
- Shape: 10000 rows x 20 cols
- Chosen target: `num_ingredients` (task=binary)
- Failure mode: **low_signal**
- Quality gate: successful_trials=1 (need >=3), best_PR_AUC=1.0000 (need >0.55)
- Root cause: likely leaky feature — PR_AUC=1.0 with only 1 trial suggests target is derivable from another column
- [ ] Improve leakage audit to catch derivative columns in pharma datasets

## 2026-04-20 04:34 UTC — FAILURE ANALYSIS

- Dataset: `abinandgkrishna/city-population` (city population)
- Detected domain: Education
- Shape: 100 rows x 4 cols
- Chosen target: `Population` (task=regression)
- Failure mode: **low_signal**
- Quality gate: successful_trials=11 (need >=3), best_R2=-0.0611 (need >0.05)
- Root cause: too few features (4 cols, 1 leaky dropped, 1 target = 2 features) and only 100 rows
- [ ] Add minimum-features guard: skip ML when usable features < 3

## 2026-04-22 06:29 UTC

- [ ] Implement text parsing for nested JSON structures in columns to extract detailed information about cast and crew.

## 2026-04-25 05:39 UTC — FAILURE ANALYSIS

- Dataset: `olistbr/brazilian-ecommerce` (Brazilian E-Commerce Public Dataset by Olist)
- Detected domain: Retail & E-commerce
- Shape: 50000 rows x 5 cols
- Chosen target: `geolocation_zip_code_prefix` (task=regression)
- Failure mode: **empty_feature_matrix**
- Detail: build_feature_matrix produced 0 usable columns. All numeric cols (lat, lng) flagged as leaky, all categoricals (city, state) dropped as high-cardinality.
- Root cause: wrong file selected — geolocation table is a lookup, not the main orders table. Dataset has multiple CSVs; kaggle_daily picked the largest by bytes, not the most analytically useful.
- [ ] Improve file selection in kaggle_discovery: prefer files with >5 columns and mixed types over large narrow tables

## Resolved

- [x] (2026-03-24) Domain mismatch HR → MDM: resolved via domain_overrides.json keywords + SKILL.md Domain Patterns
- [x] (2026-03-25) Domain mismatch Pharmaceutical → GPO Roster: resolved via domain_detector.py GPO Membership Roster domain
- [x] (2026-03-28, auto-merged 2026-04-15) Domain mismatch Retail → Financial Services: added 6 keywords to domain_overrides.json
- [x] (2026-03-31) Domain mismatch Healthcare → Healthcare RCM: resolved via domain_detector.py RCM domain
- [x] (2026-04-19) World Happiness R2=0.984 flagged as low_signal: quality gate false-negative fixed (changed OR to AND in quality gate logic)
- [x] (2026-04-24) Domain detection alignment check: addressed via existing domain detection pipeline
