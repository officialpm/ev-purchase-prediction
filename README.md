# EV purchase prediction

![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)
![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000)
![Last commit](https://img.shields.io/github/last-commit/officialpm/ev-purchase-prediction)
![Visitors](https://hits.sh/github.com/officialpm/ev-purchase-prediction.svg?label=visitors)

Copyright 2026 Parth Maniar. Licensed under Apache-2.0.

[Parth Maniar on GitHub](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/)

An S6E9 project with reproducible TE/V2 workflows for predicting electric-vehicle purchases. The best completed submission in this package scored **0.94614 public ROC AUC**, using a six-key, triple-target-encoding LightGBM model. The prior three-model V2 baseline scored **0.94563**.

## Competition and final result

**Kaggle Playground Series - Season 6 Episode 9: Predicting Electric Vehicle Purchases.**

**Final rank: 1023 / 3521. Final private ROC AUC: 0.94526.** The final leaderboard names `blend_rules.csv` as the counted entry (public 0.94613). TE was the best public entry at 0.94614, but its private score was 0.94518. Public and private winners are different.

[Competition](https://www.kaggle.com/competitions/playground-series-s6e9) | [Final leaderboard](https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard) | [Submissions](https://www.kaggle.com/competitions/playground-series-s6e9/submissions) | [Data](https://www.kaggle.com/competitions/playground-series-s6e9/data)

Final counted score rechecked October 1, 2026; final rank was verified after closure. Source/archive comparison completed October 1, 2026. Repository: [officialpm/ev-purchase-prediction](https://github.com/officialpm/ev-purchase-prediction).

### October 1 owner-supplied source

The latest supplied triple-TE archive has been inspected and merged as a preserved reference snapshot. Its training script and scored CSV are byte-for-byte identical to the TE source/artifact already included. There is no new blend source or changed model to substitute. [Comparison and hashes](docs/owner-snapshot-check.md).

### Final blend: artifact recovered, recipe not yet recovered

`predictions/final-blend.csv` is the actual downloaded scored artifact from submission 56700087. It is not a regenerated approximation. The winning source code is not in the recovered files. Its entry description says "Rank blend plus four pure training cells", but does not establish the exact formula. A tested 50/50 rank average of the saved TE LightGBM and reduced CatBoost predictions does **not** reproduce this CSV and is **not** its verified recipe. See [recovery evidence](docs/final-blend-recovery.md).

TE and V2 have recovered score-backed source code. The final blend currently has a verified prediction artifact and result, not a fully reproducible training recipe. Do not use the 50/50 experiment as if it were the final winning code.

## Results

Final scored artifacts: `final-blend.csv` public **0.94613**, private **0.94526**; `blend_candidate.csv` public 0.94612, private 0.94525 (artifact/source not included); TE public **0.94614**, private **0.94518**; V2 public 0.94563, private 0.94471. The table below separates verified local experiments from final submissions.

| Method | Five-fold OOF ROC AUC | Public ROC AUC | Status |
|---|---:|---:|---|
| V1 LightGBM / XGBoost rank blend | 0.94318 | 0.94307 | Completed historical baseline |
| V2 three-model rank blend | 0.94552 | 0.94563 | Completed baseline; code included |
| Six-key triple-TE LightGBM | **0.94593009** | **0.94614** | Best completed public submission |
| Triple-TE XGBoost | 0.94583423 | - | Local only |
| 70% TE LightGBM / 30% TE XGBoost rank blend | 0.94595876 | - | Local only; weight assessed on same OOF |
| Reduced-feature TE CatBoost | 0.94592507 | - | Completed local experiment |
| 50/50 TE LightGBM / reduced CatBoost rank blend | 0.94603517 | - | Local only; below submission gate |

The best submission improved public AUC by 0.00051 over V2. The requested 0.94945 public target has **not** been reached. OOF and public scores are different measurements; neither guarantees a private-board gain.

## Repository map

```text
notebooks/01-v2-baseline.ipynb             Portable, cleaned V2 workflow
notebooks/02-triple-target-encoding.ipynb  Portable, documented best-method workflow
src/train-v2.py                           Script version of the V2 workflow
src/train-triple-te.py                    Script version of the best-method workflow
src/build-submission.py                   Checked CSV builder
reference/scored-v2.ipynb                 Archived score-backed V2 source
reference/scored-triple-te.py             Exact original best-method training source
predictions/submission.csv               Actual TE CSV: public 0.94614, private 0.94518
predictions/final-blend.csv               Actual final CSV: public 0.94613, private 0.94526
docs/final-blend-recovery.md              Exact artifact and nonmatching recipe check
docs/owner-snapshot-check.md               October 1 supplied-code comparison
reference/owner-supplied/                  Latest owner-supplied source snapshot
results/metrics.json                      Machine-readable result ledger
WRITEUP.md                               Methodology, validation, limits and source links
requirements.txt                         Versions installed in the TE experiment environment
LICENSE                                  Apache-2.0
```

Reference files preserve historical paths. Use `src/` or `notebooks/` for local execution. Both cleaned notebooks have clear sections, empty execution outputs, and credit. They are source adaptations, not newly executed notebooks. V1 is described for context; its code is not included in this final package.

## Run locally

Use Python 3.11. Place the original S6E9 `train.csv` and `test.csv` in `data/`, preserving row order. Competition data is **not** bundled. Install dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train-triple-te.py
python src/build-submission.py --test data/test.csv --predictions data/triple_te_compact_test.npy --out predictions/submission-rerun.csv
```

For another data folder, set `EV_DATA_DIR=/absolute/path/to/data`. Scripts and notebooks should run from the repository root. To run V2 instead:

```bash
python src/train-v2.py
```

V2 writes `predictions/v2-submission.csv` and may need much more memory than the TE model because every original feature is also represented as a CatBoost string category. Training takes hardware-dependent time. Nothing installs paid services, uploads predictions, or submits automatically. The pinned versions were inspected in the TE run environment; the historical hosted V2 environment was not fully recorded. Version changes can affect rerun scores.

## What the best method does

1. Remove the target and ID from predictors; map `Will_Buy_EV == "Yes"` to 1.
2. Keep original features, native categorical columns, and label-free count columns derived from train + test feature distributions.
3. Make six integer keys: exact value, floor(value / 100), floor(value / 1000), for both annual income and daily commute.
4. For each key, add three target-encoding columns with smoothing `auto`, 10 and 100: 18 columns in total.
5. Use five stratified outer folds and five stratified inner folds to cross-fit training encodings. Outer validation and test encodings use only outer-training targets.
6. Fit LightGBM with fold-level early stopping. Average five test vectors and write one probability per test ID.

**Important:** `auto` is a compact empirical-Bayes approximation, **not** scikit-learn's exact `TargetEncoder(smooth="auto")`. The exact-auto replacement was prepared but not run when this package was closed. See the formula and validation limits in [WRITEUP.md](WRITEUP.md).

## What is not in this repository

Competition `train.csv` and `test.csv` are not bundled. Put them in `data/` locally. `.gitignore` also keeps virtualenvs, caches, `.npy` outputs, logs, and `.env` files out of git. The two prediction CSVs that are included are scored artifacts, not training data and not labels. This package contains no API tokens or credentials.

## Credit

Code and documentation: Parth Maniar, Copyright 2026, Apache-2.0. Competition data and rules belong to their organizer. NumPy, pandas, SciPy, scikit-learn, LightGBM, XGBoost and CatBoost retain their own licenses. No third-party public notebook implementation is claimed as original work. [Detailed source links and provenance](WRITEUP.md#sources-and-provenance).

Badges: visitor count is third-party page hits, not unique people. License, Python and Black badges describe this package; no unrun test-status badge is shown. The last-commit badge points at this GitHub repository.
