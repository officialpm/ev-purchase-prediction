# EV purchase prediction: S6E9 methods and final results

Copyright 2026 Parth Maniar. Licensed under Apache-2.0.

**Author:** [Parth Maniar](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/)

## Final competition result

Kaggle Playground Series S6E9 - Predicting Electric Vehicle Purchases closed September 30, 2026. Final rank: **1023 / 3521**. Final counted entry: `blend_rules.csv`, private ROC AUC **0.94526**, public **0.94613**. The best public entry was triple-TE LightGBM at **0.94614**, private **0.94518**. V2 private score was **0.94471**.

The official final leaderboard and own submissions page were read after closure. Automatic selection evaluated the public-best entries, then the higher evaluated private score counted. No post-close selection change was made.

The actual final blend CSV is included, but its source recipe remains unrecovered. It cannot be reconstructed from the matching rounded OOF in a page description. The saved 50/50 rank experiment fails both exact and rank-equivalence checks against the scored CSV. [Recovery evidence](docs/final-blend-recovery.md) preserves the measured mismatch. No guessed recipe is presented as the winning code.

## Summary

The best completed public submission reached **0.94614 ROC AUC**. Its five-fold out-of-fold AUC was **0.9459300948605641**. It beat the prior V2 public result of 0.94563 by 0.00051. This report closes the documented score-backed work, not a claim that every follow-up experiment has completed. The 0.94945 public target remains unmet.

The prediction problem is binary: estimate purchase propensity `Will_Buy_EV` from synthetic customer attributes. There are 668,665 training rows and 286,571 test rows. Map `Yes` to 1. Exclude `id` from predictors. Evaluation is ROC AUC, so use continuous scores rather than thresholded labels.

## Baseline: V2

V2 fits LightGBM, XGBoost and CatBoost with five stratified folds, shuffled with seed 1120. LightGBM and XGBoost use original features plus count columns. CatBoost uses every original feature as a string category and retains numeric features in numeric form too.

Each model creates OOF predictions and an average test prediction. A simplex grid in steps of 0.05 chooses a rank blend. The scored run chose **0% LightGBM, 5% XGBoost, 95% CatBoost**. Thus "three-model blend" describes the search, not three positive final weights. V2 reached 0.94552 OOF and 0.94563 public AUC.

Blend weights were selected on the same OOF labels used to report blend AUC. That value can be optimistic. There is no independent second holdout for blend selection. The cleaned V2 code fixes one misleading intermediate print: it no longer reports AUC for the still-zero CatBoost vector before CatBoost training. Training logic is otherwise retained, with local input/output paths.

## Best method: six keys, three smoothing values

For `Annual_Income_USD` and `Daily_Commute_km`, define three keys each:

- Integer key: `floor(x)`.
- Medium key: `floor(x / 100)`.
- Coarse key: `floor(x / 1000)`.

That gives six keys. Missing numeric values use the original script's sentinel before flooring. The coarse commute bins may collapse many values into the same key; the implementation retains this requested resolution rather than silently changing it.

Each key receives three target encodings: `auto`, 10 and 100. For fixed smoothing `s`, category `k` has:

```text
encoding(k) = (sum of category targets + s * training prior) / (category count + s)
```

Unknown categories fall back to the applicable training prior. The compact `auto` approximation uses observed category proportions `p_k`:

```text
between = max(sample variance of p_k, 1e-6)
within  = max(mean(p_k * (1 - p_k)), 1e-6)
lambda  = max(1, within / between)
encoding(k) = (target_sum_k + lambda * prior) / (count_k + lambda)
```

This is not the exact scikit-learn encoder. Do not label the scored method as exact `TargetEncoder(smooth="auto")`. A separate exact implementation was prepared later; it has no measured result in this package.

Original features and train + test count columns remain in the model. Counts use unlabeled test-feature distributions, a transductive preprocessing choice. No test targets are available or used.

## Validation and leakage controls

- Outer validation: `StratifiedKFold(5, shuffle=True, random_state=1120)`.
- Inner training encodings: five stratified folds with seed `1120 + outer_fold`.
- A training row's encoding is calculated without its own inner-fold labels.
- Outer-validation encodings use outer-training labels only.
- Test encodings use outer-training labels only.
- The final test prediction averages the five outer-fold models.

These controls prevent own-target encoding and outer-validation target leakage. They do not eliminate optimism from early stopping on the outer validation fold or from comparing many variants against the same OOF split. Results are development validation, not an untouched final test.

LightGBM settings: learning rate 0.03; up to 4,500 estimators; 31 leaves; minimum child samples 60; feature fraction 0.8; row fraction 0.8 with frequency 1; L2 regularization 3; seed 1120; three worker threads; early-stopping patience 100.

## Measured results

| Method | OOF AUC | Public AUC |
|---|---:|---:|
| Historical V1 LGBM/XGB rank blend | 0.94318 | 0.94307 |
| V2 blend, 95% CatBoost / 5% XGBoost | 0.94552 | 0.94563 |
| Best triple-TE LightGBM | 0.94593009 | **0.94614** |
| TE-augmented XGBoost | 0.94583423 | Not submitted |
| 70/30 LGBM/XGB rank blend | 0.94595876 | Not submitted |
| Reduced-feature TE CatBoost | 0.94592507 | Not submitted |
| 50/50 TE LightGBM / reduced CatBoost rank blend | 0.94603517 | Not submitted |

Best-method fold AUCs, in order: 0.94612480, 0.94488119, 0.94653788, 0.94647012, 0.94567979. Aggregate OOF AUC is computed over all OOF rows, not the arithmetic mean of these five values.

The 70/30 rank blend gained about 0.000029 over standalone TE LightGBM on the same OOF used to assess its weight. This is too small to establish a public improvement, and it did not clear the 0.94614 development submission gate. A previous local feature-engineering attempt also failed to beat its baseline and was not submitted.

Full-feature TE CatBoost stopped without a completed fold; memory pressure was suspected but not proven. A reduced-feature CatBoost alternative completed five folds with OOF AUC 0.94592507. It is not an exact V2 CatBoost feature view. Fixed rank blends scored 0.94594363 at the prior V2 weights (0/5/95), 0.94603517 for equal LightGBM/CatBoost, and 0.94602880 for an equal three-model blend. None of these local scores cleared 0.94614. Post-close inspection nevertheless found two later scored entries, `blend_rules.csv` and `blend_candidate.csv`; their exact source is not recovered. Their page descriptions are not proof that they came from these local experiments. Stacking and seed averaging have no tested result to report here. They are not presented as improvements.

## October 1 source reconciliation

The owner supplied the triple-TE snapshot again for this final handoff. Byte-level comparison confirmed that its training source and scored prediction CSV match the already recovered best-public TE version exactly. Both supplied scripts, README and license are preserved unchanged under `reference/owner-supplied/`; the identical large CSV is included once. [Comparison evidence](docs/owner-snapshot-check.md). This archive does not recover the winning final-blend recipe, and no guessed merge replaces it.

## Artifact and reproduction

`predictions/submission.csv` is the actual completed 0.94614 CSV: 286,571 rows, columns `id,Will_Buy_EV`, unique IDs, finite scores in [0, 1]. Its SHA256 is:

```text
339036af14e1ed28e566bd9be92dc6b192b54a7f8cd15a825c28643613abddef
```

The exact original TE source is preserved in `reference/scored-triple-te.py`. Its SHA256 is:

```text
b43c928b2af86f37a6c2be24fe65438c7e8667ced61d5d8d5ac2cefb33a205d0
```

Portable scripts change the data path and provide local output handling. Cleaned notebooks split the best-method workflow into readable sections. These adaptations were syntax-checked and the CSV builder was tested against the saved vector; neither full training pipeline was retrained for packaging. Outputs are intentionally cleared. Exact package versions for the local TE environment are pinned, but historical V2 runtime versions are not fully known. Data is not redistributed.

## Limits

The public leaderboard is a hidden subset, not the final private leaderboard. The final rank and private scores above were verified after closure; the exact final-blend recipe remains a reproduction limit. Synthetic-data patterns may not transfer to real customer behavior. Count encoding includes unlabeled test distribution. Memory and runtime vary; V2's all-column categorical representation can be expensive. Small OOF gains and weight searches on reused folds need cautious interpretation.

The competition deadline is September 30, 2026, 11:59 PM UTC / 4:59 PM PDT. At the last score-backed selection check, fewer than two finalists were selected and the site stated that it automatically chooses best-scoring submissions. The deadline has passed and final standings are now visible.

## Sources and provenance

- [Competition](https://www.kaggle.com/competitions/playground-series-s6e9).
- [Final leaderboard](https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard).
- [Competition data](https://www.kaggle.com/competitions/playground-series-s6e9/data).
- [Completed submissions and selection state](https://www.kaggle.com/competitions/playground-series-s6e9/submissions).
- [Score-backed V1 source](https://www.kaggle.com/code/theteacoder/ev-purchase-lgbm-xgb-ensemble-parth-maniar?scriptVersionId=352401500).
- [Score-backed V2 source](https://www.kaggle.com/code/theteacoder/ev-purchase-lgbm-xgb-ensemble-parth-maniar?scriptVersionId=352473704).

The source version links are private to the account owner; they are provenance, not promises of public accessibility. V2's archived source was recovered from the scored version. The six-key experiment follows Parth Maniar's requested method, with the auto-smoothing approximation disclosed. No third-party public notebook code was copied for this method. Credit the modeling libraries and the competition organizer separately from this project's author.

Code and documentation Copyright 2026 Parth Maniar, Apache-2.0. [GitHub](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/).
