# EV purchase: multi-resolution keys and triple target encoding

Copyright 2026 Parth Maniar. Licensed under Apache-2.0 (see LICENSE).

Parth Maniar: [GitHub](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/)

This snapshot is the memory-bounded standalone LightGBM experiment behind the scored S6E9 CSV. It adds six resolution keys for `Annual_Income_USD` and `Daily_Commute_km`: integer value, floor(value / 100), and floor(value / 1000). Each key gets three cross-fitted target encodings with smoothing `auto` (approximation), 10, and 100. Original columns plus transductive count features are kept. Five stratified folds, with inner cross-fitting for training encodings, avoid encoding a row with its own target. Validation and test keys are encoded from each outer training fold only.

## Results, not promises

- Five-fold local OOF ROC AUC: **0.9459300949** on 668,665 rows.
- The submitted `predictions/submission.csv` completed with **0.94614 public ROC AUC** on September 29, 2026. These scores are not interchangeable; public score can vary on the hidden test partition.
- Prior scored three-model V2 public ROC AUC was 0.94563. This snapshot does **not** contain V2's three-model blend or any in-progress XGBoost experiments.
- **Important:** `smooth='auto'` in this script is a compact empirical-Bayes moment-shrinkage approximation. It is **not** scikit-learn's exact `TargetEncoder(smooth='auto')` formula. A first exact encoder attempt exceeded the available memory, so do not report this run as using exact auto smoothing.

## Run

Use Python 3.11 with `numpy`, `pandas`, `scikit-learn`, and `lightgbm`. From the directory containing this README, make `data/` and place the S6E9 competition's `train.csv` and `test.csv` there. No data or credentials are bundled. The script's `P` constant currently points to `/home/sandbox/s6e9/data/`; change it to the absolute location of your local `data/` directory before running:

```bash
python src/triple-te-compact.py
python src/build-submission.py --test data/test.csv --predictions data/triple_te_compact_test.npy --out predictions/submission.csv
```

The experiment writes `triple_te_compact_oof.npy`, `triple_te_compact_test.npy`, and a partial OOF checkpoint into that data directory. It can take hours depending on hardware. Memory use was designed for a small worker, but your runtime will vary. Keep the original training/test row order. The supplied prediction CSV is the actual scored file, with 286,571 rows and columns `id,Will_Buy_EV`; it is included so you can compare independently without rerunning first.

The code uses only the competition data and standard open-source packages, with no borrowed public notebook implementation. The open-source package authors retain their respective copyrights and licenses.
