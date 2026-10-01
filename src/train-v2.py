# Copyright 2026 Parth Maniar. Apache-2.0.
# Copyright 2026 Parth Maniar. Licensed under the Apache License, Version 2.0.
import os, glob, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, xgboost as xgb

warnings.filterwarnings("ignore")
root = os.environ.get("EV_DATA_DIR", "data")
train = pd.read_csv(f"{root}/train.csv")
test = pd.read_csv(f"{root}/test.csv")
TARGET, ID = "Will_Buy_EV", "id"
print(train.shape, test.shape)
if train[TARGET].dtype == object:
    classes = sorted(train[TARGET].unique())
    print("target classes:", classes)
    pos = "Yes" if "Yes" in classes else classes[-1]
    train[TARGET] = (train[TARGET] == pos).astype(int)
print("target rate:", round(train[TARGET].mean(), 4))
print(
    pd.DataFrame(
        {
            "dtype": train.dtypes.astype(str),
            "nunique": train.nunique(),
            "missing": train.isna().mean().round(4),
        }
    )
)


# Copyright 2026 Parth Maniar. Apache-2.0.
features = [c for c in train.columns if c not in (TARGET, ID)]
cat_cols = [
    c for c in features if train[c].dtype == object or str(train[c].dtype) == "category"
]
full = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
for c in cat_cols:
    full[c] = full[c].astype(str).astype("category").cat.codes
for c in features:  # count encoding over train+test
    full[c + "_cnt"] = full[c].map(full[c].value_counts()).astype("float32")
for c in cat_cols:
    full[c] = full[c].astype("category")
X = full.iloc[: len(train)].reset_index(drop=True)
X_test = full.iloc[len(train) :].reset_index(drop=True)
y = train[TARGET].values
print("features:", X.shape[1], "| categorical:", cat_cols)


# Copyright 2026 Parth Maniar. Apache-2.0.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1120)
oof = {"lgb": np.zeros(len(X)), "xgb": np.zeros(len(X)), "cat": np.zeros(len(X))}
pred = {
    "lgb": np.zeros(len(X_test)),
    "xgb": np.zeros(len(X_test)),
    "cat": np.zeros(len(X_test)),
}
lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.02,
    num_leaves=63,
    min_child_samples=50,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    verbose=-1,
    seed=1120,
    n_jobs=-1,
)
xgb_params = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    seed=1120,
    nthread=-1,
)
dtest = xgb.DMatrix(X_test, enable_categorical=True)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    t = time.time()
    m = lgb.train(
        lgb_params,
        lgb.Dataset(X.iloc[tr], y[tr]),
        8000,
        valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    oof["lgb"][va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pred["lgb"] += m.predict(X_test, num_iteration=m.best_iteration) / 5
    dtr = xgb.DMatrix(X.iloc[tr], y[tr], enable_categorical=True)
    dva = xgb.DMatrix(X.iloc[va], y[va], enable_categorical=True)
    b = xgb.train(
        xgb_params,
        dtr,
        8000,
        evals=[(dva, "va")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    oof["xgb"][va] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
    pred["xgb"] += b.predict(dtest, iteration_range=(0, b.best_iteration + 1)) / 5
    print(
        f"fold {fold}: lgb {roc_auc_score(y[va], oof['lgb'][va]):.5f} "
        f"| xgb {roc_auc_score(y[va], oof['xgb'][va]):.5f} "
        f"| {time.time()-t:.0f}s"
    )
for k in ("lgb", "xgb"):
    print(k, "OOF AUC", round(roc_auc_score(y, oof[k]), 5))


# Copyright 2026 Parth Maniar. Apache-2.0.
# CatBoost: all original columns as categoricals (as strings) plus the raw numerics.
from catboost import CatBoostClassifier, Pool

raw = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
C = pd.DataFrame(index=raw.index)
cb_cats = []
for c in features:
    C[c + "_c"] = raw[c].astype(str)
    cb_cats.append(c + "_c")
    if c not in cat_cols:
        C[c] = raw[c].astype("float32")
Cx, Ct = C.iloc[: len(train)].reset_index(drop=True), C.iloc[len(train) :].reset_index(
    drop=True
)
test_pool = Pool(Ct, cat_features=cb_cats)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    t = time.time()
    m = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=150,
        random_seed=1120,
        verbose=0,
        thread_count=-1,
    )
    m.fit(
        Pool(Cx.iloc[tr], y[tr], cat_features=cb_cats),
        eval_set=Pool(Cx.iloc[va], y[va], cat_features=cb_cats),
        use_best_model=True,
    )
    oof["cat"][va] = m.predict_proba(Cx.iloc[va])[:, 1]
    pred["cat"] += m.predict_proba(test_pool)[:, 1] / 5
    print(
        f"fold {fold}: cat {roc_auc_score(y[va], oof['cat'][va]):.5f} "
        f"| it {m.get_best_iteration()} | {time.time()-t:.0f}s"
    )
print("cat OOF AUC", round(roc_auc_score(y, oof["cat"]), 5))


# Copyright 2026 Parth Maniar. Apache-2.0.
from scipy.stats import rankdata

r = lambda a: rankdata(a) / len(a)
R = {k: r(v) for k, v in oof.items()}
RT = {k: r(v) for k, v in pred.items()}
# coarse grid over simplex weights (step 0.05)
best, best_auc = None, 0
for a in np.arange(0, 1.0001, 0.05):
    for b in np.arange(0, 1.0001 - a, 0.05):
        w = {"lgb": a, "xgb": b, "cat": 1 - a - b}
        auc = roc_auc_score(y, sum(w[k] * R[k] for k in R))
        if auc > best_auc:
            best, best_auc = w, auc
print(
    "weights", {k: round(v, 2) for k, v in best.items()}, "OOF AUC", round(best_auc, 5)
)
sub = pd.DataFrame({ID: test[ID], TARGET: sum(best[k] * RT[k] for k in RT)})
os.makedirs("predictions", exist_ok=True)
sub.to_csv("predictions/v2-submission.csv", index=False)
print(sub.shape)
print(sub.head())
