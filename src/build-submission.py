# Copyright 2026 Parth Maniar. Licensed under Apache-2.0.
"""Build a prediction CSV from the saved test vector and original test IDs."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--test", default="data/test.csv")
p.add_argument("--predictions", default="data/triple_te_compact_test.npy")
p.add_argument("--out", default="predictions/submission.csv")
a = p.parse_args()
test = pd.read_csv(a.test, usecols=["id"])
pred = np.load(a.predictions)
assert len(test) == len(pred) and test["id"].is_unique
assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": test.id, "Will_Buy_EV": pred}).to_csv(a.out, index=False)
print("Wrote", a.out, len(pred), "rows")
