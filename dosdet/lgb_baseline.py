from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit("Please: pip install lightgbm")

def load_parquet_cache(cache_dir: str):
    manifests = json.load(open(os.path.join(cache_dir, "manifest.json")))
    Xs, y, files, t0s = [], [], [], []
    for shard in manifests["files"]:
        df = pd.read_parquet(shard["path"])
        # static only
        Xs.append(np.stack(df["static"].tolist()))
        y.append(df["y"].values.astype(int))
        files += df["file"].tolist()
        t0s += df["t0"].tolist()
    X = np.vstack(Xs); Y = np.concatenate(y)
    return X, Y, files, np.array(t0s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cache_dir = cfg["paths"]["cache_dir"]

    X, Y, files, t0 = load_parquet_cache(cache_dir)
    Xtr, Xva, ytr, yva = train_test_split(X, Y, test_size=0.2, random_state=1337, stratify=Y)
    train_data = lgb.Dataset(Xtr, label=ytr)
    val_data = lgb.Dataset(Xva, label=yva)
    params = dict(objective="binary", metric=["auc","average_precision"], learning_rate=0.05,
                  num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)
    gbm = lgb.train(params, train_data, num_boost_round=800, valid_sets=[val_data], verbose_eval=50, early_stopping_rounds=50)
    proba = gbm.predict(Xva, num_iteration=gbm.best_iteration)
    print("LightGBM ROC-AUC:", roc_auc_score(yva, proba), "PR-AUC:", average_precision_score(yva, proba))

    # Save teacher logits (logit = log(p/(1-p)))
    eps=1e-6
    logits = np.log(np.clip(proba,eps,1-eps) / np.clip(1-proba,eps,1-eps))
    out = pd.DataFrame({"file": files[:len(Xva)], "t0": t0[:len(Xva)], "logit": logits})
    # We want all rows; recompute over whole X
    proba_all = gbm.predict(X, num_iteration=gbm.best_iteration)
    logits_all = np.log(np.clip(proba_all,eps,1-eps) / np.clip(1-proba_all,eps,1-eps))
    df_all = pd.DataFrame({"file": files, "t0": t0, "logit": logits_all})
    df_all.to_parquet(os.path.join(cache_dir, "lgbm_teacher_logits.parquet"), index=False)

    # Save importances
    imp = gbm.feature_importance(importance_type="gain")
    with open(os.path.join(cache_dir, "lgbm_importances.json"), "w") as f:
        json.dump({"importances": imp.tolist()}, f, indent=2)

if __name__ == "__main__":
    main()
