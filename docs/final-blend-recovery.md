# Final blend recovery

Copyright 2026 Parth Maniar. Apache-2.0.

## Verified artifact

- Own submission ID: 56700087, uploaded filename `blend_rules.csv`.
- Download: own submissions page, authenticated read of its uploaded-file download resource.
- Artifact: `predictions/final-blend.csv`, 5,051,446 bytes, 286,571 rows.
- Columns: `id,Will_Buy_EV`. IDs verified against the original test file.
- Public ROC AUC: 0.94613. Private ROC AUC: 0.94526. Final rank: 1023 / 3521.
- SHA256: `4dc6068a601d9ac94b7743fdc527de9b09340f3e78cabab43086d2e6a97c491b`.
- [Submission evidence](https://www.kaggle.com/competitions/playground-series-s6e9/submissions) and [final leaderboard](https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard).

## Nonmatching experiment

The previously measured local 50/50 TE LightGBM / reduced CatBoost rank blend had OOF ROC AUC 0.94603516848509. A matching rounded description value does not establish provenance. Its test recipe was tested as:

```python
(rankdata(te_lgb_test) + rankdata(reduced_cat_test)) / (2 * len(test))
```

Compared with the downloaded winning CSV:

- Mean absolute score difference: 0.004974874428821487.
- Maximum absolute score difference: 0.7147199117751831.
- Spearman rank correlation: 0.9987531959728686, not 1.
- 286,513 rows differ by more than 1e-7.

Thus it is neither exact nor rank-equivalent. The winning description also mentions four pure training cells; their thresholds, rules, and implementation are not verified. Do not infer them or claim the 50/50 recipe reproduces the final winner. No guessed winning script is included. This section will be revised only if exact source or verifiable reconstruction is recovered.
