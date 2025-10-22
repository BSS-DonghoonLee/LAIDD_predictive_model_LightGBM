%%bash
cat > LightGBM_Optuna.py << 'PYEOS'
# =========================================================
# LightGBM + Optuna (ECFP_bin) + 5-Fold Ensemble Training
# =========================================================
import os, json, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless 환경에서 그림 저장
import matplotlib.pyplot as plt
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import lightgbm as lgb
from lightgbm import LGBMRegressor

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# Fingerprint builders
# --------------------------
def ecfp_binary_array(mol, radius=3, n_bits=4096, use_chirality=True, use_features=False):
    bv = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits,
        useChirality=use_chirality, useFeatures=use_features
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def build_ecfp_bin(smiles_list, radius=3, n_bits=4096):
    xb_list = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            xb_list.append(np.zeros(n_bits, dtype=np.float32))
        else:
            xb_list.append(ecfp_binary_array(m, radius=radius, n_bits=n_bits))
    XB = np.vstack(xb_list)
    return XB

def main():
    # --------------------------
    # Args
    # --------------------------
    p = argparse.ArgumentParser(description="Train LightGBM with Optuna on ECFP (binary) and save 5-fold ensemble.")
    p.add_argument("--csv", required=True, help="CSV path (must have SMILES and target columns)")
    p.add_argument("--smiles", default="Smiles", help="SMILES column name")
    p.add_argument("--target", default="pIC50", help="Target column name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--ecfp-radius", type=int, default=2)
    p.add_argument("--ecfp-bits", type=int, default=2048)
    p.add_argument("--out", default="./checkpoints_lgbm_ensemble", help="output dir for checkpoints & artifacts")
    args = p.parse_args()

    # --------------------------
    # IO & seed
    # --------------------------
    csv_path   = args.csv
    smiles_col = args.smiles
    target_col = args.target
    seed       = args.seed
    test_size  = args.test_size
    n_splits   = args.splits
    patience   = args.patience
    n_trials   = args.n_trials
    timeout    = args.timeout
    ecfp_radius= args.ecfp_radius
    ecfp_bits  = args.ecfp_bits

    ckpt_dir = args.out
    os.makedirs(ckpt_dir, exist_ok=True)
    np.random.seed(seed)

    # --------------------------
    # Load data & featurize
    # --------------------------
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columns not found. Need '{smiles_col}' and '{target_col}'. CSV has: {list(df.columns)}")

    smiles = df[smiles_col].astype(str).tolist()
    y_all  = df[target_col].to_numpy(dtype=np.float32)

    XB = build_ecfp_bin(smiles, radius=ecfp_radius, n_bits=ecfp_bits)

    # --------------------------
    # Holdout split
    # --------------------------
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        XB, y_all, test_size=test_size, random_state=seed, shuffle=True
    )

    # --------------------------
    # Optuna objective (minimize mean Val MSE on KFold)
    # --------------------------
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_categorical("max_depth", [-1] + list(range(3, 17))),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "objective": "regression",
            "verbosity": -1,
            "random_state": seed,
            "n_jobs": -1,
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        val_mses = []

        for fold_idx, (tr, va) in enumerate(kf.split(X_train_all), start=1):
            X_tr, X_val = X_train_all[tr], X_train_all[va]
            y_tr, y_val = y_train_all[tr], y_train_all[va]

            model = LGBMRegressor(**params)

            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)]
                )
            except TypeError:
                try:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric="l2",
                        early_stopping_rounds=patience
                    )
                except TypeError:
                    model.fit(X_tr, y_tr)

            y_val_pred = model.predict(X_val)
            v_mse = mean_squared_error(y_val, y_val_pred)
            val_mses.append(v_mse)

            trial.report(float(np.mean(val_mses)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(val_mses))

    # --------------------------
    # Run study
    # --------------------------
    sampler = TPESampler(seed=seed)
    pruner  = MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print("\n=== Best Trial ===")
    print("Value (CV Val MSE):", study.best_trial.value)
    print("Params:", study.best_trial.params)

    best_params = {
        **study.best_trial.params,
        "objective": "regression",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
    }
    with open(os.path.join(ckpt_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(ckpt_dir, "optuna_best.txt"), "w") as f:
        f.write(f"best_value={study.best_trial.value}\n")
        f.write(str(study.best_trial.params))

    # --------------------------
    # 5-fold train & save
    # --------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = []
    oof_pred = np.zeros_like(y_train_all, dtype=float)
    fold_test_preds = []

    val_mse_list, val_pcc_list = [], []
    test_mse_list, test_pcc_list = [], []

    for fold, (tr, va) in enumerate(kf.split(X_train_all), start=1):
        X_tr, X_val = X_train_all[tr], X_train_all[va]
        y_tr, y_val = y_train_all[tr], y_train_all[va]

        m = LGBMRegressor(**best_params)
        try:
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)]
            )
        except TypeError:
            try:
                m.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="l2",
                    early_stopping_rounds=patience
                )
            except TypeError:
                m.fit(X_tr, y_tr)

        models.append(m)

        booster = m.booster_
        best_iter = getattr(m, "best_iteration_", None)
        booster.save_model(os.path.join(ckpt_dir, f"model_fold{fold}_best.txt"),
                           num_iteration=best_iter)
        joblib.dump(m, os.path.join(ckpt_dir, f"model_fold{fold}.pkl"))

        # OOF
        oof_pred[va] = m.predict(X_val)
        v_mse = mean_squared_error(y_val, oof_pred[va])
        v_pcc = pearsonr(y_val, oof_pred[va])[0] if len(y_val) > 1 else np.nan
        val_mse_list.append(v_mse)
        val_pcc_list.append(v_pcc)

        # test pred per fold
        fold_test_pred = m.predict(X_test)
        fold_test_preds.append(fold_test_pred)

        t_mse = mean_squared_error(y_test, fold_test_pred)
        t_pcc = pearsonr(y_test, fold_test_pred)[0] if len(y_test) > 1 else np.nan
        test_mse_list.append(t_mse)
        test_pcc_list.append(t_pcc)

    # --------------------------
    # Ensemble & metrics
    # --------------------------
    y_test_pred_ens = np.mean(fold_test_preds, axis=0)

    oof_mse  = mean_squared_error(y_train_all, oof_pred)
    oof_pcc  = pearsonr(y_train_all, oof_pred)[0] if len(y_train_all) > 1 else np.nan
    ens_mse  = mean_squared_error(y_test, y_test_pred_ens)
    ens_pcc  = pearsonr(y_test, y_test_pred_ens)[0] if len(y_test) > 1 else np.nan

    print("\n=== CV(5) Diagnostics ===")
    print(f"Fold Val MSE mean±std: {np.mean(val_mse_list):.4f} ± {np.std(val_mse_list):.4f}")
    print(f"Fold Val PCC mean±std: {np.mean(val_pcc_list):.4f} ± {np.std(val_pcc_list):.4f}")
    print(f"OOF MSE: {oof_mse:.4f}, OOF PCC: {oof_pcc:.4f}")

    print("\n=== Individual Fold Models on Test (reference) ===")
    print(f"Test MSE mean±std: {np.mean(test_mse_list):.4f} ± {np.std(test_mse_list):.4f}")
    print(f"Test PCC mean±std: {np.mean(test_pcc_list):.4f} ± {np.std(test_pcc_list):.4f}")

    print("\n=== Ensemble (Mean of 5 folds) on Test ===")
    print(f"Ensemble Test MSE: {ens_mse:.4f}")
    print(f"Ensemble Test PCC: {ens_pcc:.4f}")

    # save artifacts
    np.save(os.path.join(ckpt_dir, "oof_pred.npy"), oof_pred)
    np.save(os.path.join(ckpt_dir, "y_train_all.npy"), y_train_all)
    np.save(os.path.join(ckpt_dir, "y_test.npy"), y_test)
    np.save(os.path.join(ckpt_dir, "y_test_pred_ens.npy"), y_test_pred_ens)

    metrics = {
        "oof_mse": float(oof_mse),
        "oof_pcc": float(oof_pcc),
        "ens_mse": float(ens_mse),
        "ens_pcc": float(ens_pcc),
        "val_mse_mean": float(np.mean(val_mse_list)),
        "val_mse_std":  float(np.std(val_mse_list)),
        "val_pcc_mean": float(np.mean(val_pcc_list)),
        "val_pcc_std":  float(np.std(val_pcc_list)),
        "test_mse_mean": float(np.mean(test_mse_list)),
        "test_mse_std":  float(np.std(test_mse_list)),
        "test_pcc_mean": float(np.mean(test_pcc_list)),
        "test_pcc_std":  float(np.std(test_pcc_list)),
        "n_splits": n_splits,
        "ecfp_radius": ecfp_radius,
        "ecfp_bits": ecfp_bits,
        "seed": seed,
        "test_size": test_size,
    }
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --------------------------
    # Scatter plot
    # --------------------------
    y_all_min = float(min(y_train_all.min(), y_test.min()))
    y_all_max = float(max(y_train_all.max(), y_test.max()))
    pad = 0.05 * (y_all_max - y_all_min if y_all_max > y_all_min else 1.0)
    xmin, xmax = y_all_min - pad, y_all_max + pad
    ymin, ymax = xmin, xmax

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred_ens, alpha=0.7)
    plt.plot([xmin, xmax], [ymin, ymax], linestyle='--', color='red')
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.xlabel("True (Test)")
    plt.ylabel("Predicted (Test)")
    plt.title(f'LightGBM+Optuna – 5Fold Ensemble\nTest MSE={ens_mse:.3f}, PCC={ens_pcc:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "scatter_test_ensemble.png"), dpi=200)

    print(f"\nArtifacts saved to: {os.path.abspath(ckpt_dir)}")

# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    main()
PYEOS
