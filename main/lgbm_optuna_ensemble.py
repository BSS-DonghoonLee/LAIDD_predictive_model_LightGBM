# =========================================================
# LGBM + Optuna 튜닝 (ECFP_bin 단일 피처셋) + 5-Fold 앙상블 학습/저장
# =========================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # scikit-learn 설치 시 함께 포함되는 경우가 많음

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys  # MACCS는 안쓰지만 RDKit 설치 확인 겸 유지

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import lightgbm as lgb
from lightgbm import LGBMRegressor

# --- Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# --------------------------
# 사용자 설정
# --------------------------
csv_path   = '/content/EGFR_T790M_L858R.csv'   # 데이터 경로
smiles_col = 'Smiles'
target_col = 'pIC50'

seed      = 42
test_size = 0.2
n_splits  = 5
patience  = 100

# Optuna 설정
n_trials  = 200   # 탐색 횟수 (필요시 늘리세요)
timeout   = None # 시간 제한(초). 원하시면 예: 3600

# Morgan 설정 (ECFP)
ecfp_radius = 2
ecfp_bits   = 2048

# 체크포인트 저장 디렉토리
ckpt_dir = "./checkpoints_lgbm_ensemble(T790M_L858R)"
os.makedirs(ckpt_dir, exist_ok=True)

# --------------------------
# 피처 빌더 (ECFP bin만 사용)
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

# --------------------------
# 데이터 로딩 & 피처 준비 (ECFP_bin)
# --------------------------
df = pd.read_csv(csv_path)
smiles = df[smiles_col].astype(str).tolist()
y_all  = df[target_col].to_numpy(dtype=np.float32)

# 기본 배열 한번만 생성 (ECFP_bin)
XB = build_ecfp_bin(smiles, radius=ecfp_radius, n_bits=ecfp_bits)

# --------------------------
# 홀드아웃 분할 (테스트 세트 고정)
# --------------------------
X_train_all, X_test, y_train_all, y_test = train_test_split(
    XB, y_all, test_size=test_size, random_state=seed, shuffle=True
)

# --------------------------
# Optuna 목적함수 (KFold 평균 Val MSE 최소화)
# --------------------------
def objective(trial: optuna.Trial) -> float:
    # 하이퍼파라미터 탐색 공간
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        # max_depth: -1(무제한)도 포함
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

        # 조기종료(ES) 적용
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

        # 중간값 보고 + 프루닝 체크
        trial.report(np.mean(val_mses), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(val_mses))

# --------------------------
# Optuna Study 실행
# --------------------------
sampler = TPESampler(seed=seed)
pruner  = MedianPruner(n_warmup_steps=2)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

print("
=== Best Trial ===")
print("Value (CV Val MSE):", study.best_trial.value)
print("Params:", study.best_trial.params)

# 최적 파라미터 저장
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
    f.write(f"best_value={study.best_trial.value}
")
    f.write(str(study.best_trial.params))

# --------------------------
# 5-Fold CV 앙상블 학습 + 모델(5개) 저장
# --------------------------
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
models = []
oof_pred = np.zeros_like(y_train_all, dtype=float)  # OOF 진단용
fold_test_preds = []                                 # 각 폴드 모델의 테스트 예측

val_mse_list, val_pcc_list = [], []
test_mse_list, test_pcc_list = [], []

for fold, (tr, va) in enumerate(kf.split(X_train_all), start=1):
    X_tr, X_val = X_train_all[tr], X_train_all[va]
    y_tr, y_val = y_train_all[tr], y_train_all[va]

    m = LGBMRegressor(**best_params)
    # ES는 폴드의 검증셋으로
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

    # --- 폴드 모델 저장 (텍스트 ckpt + sklearn wrapper pkl)
    booster = m.booster_
    best_iter = getattr(m, "best_iteration_", None)
    txt_path = os.path.join(ckpt_dir, f"model_fold{fold}_best.txt")
    booster.save_model(txt_path, num_iteration=best_iter)

    pkl_path = os.path.join(ckpt_dir, f"model_fold{fold}.pkl")
    joblib.dump(m, pkl_path)

    # OOF
    oof_pred[va] = m.predict(X_val)
    v_mse = mean_squared_error(y_val, oof_pred[va])
    v_pcc = pearsonr(y_val, oof_pred[va])[0] if len(y_val) > 1 else np.nan
    val_mse_list.append(v_mse)
    val_pcc_list.append(v_pcc)

    # 개별 모델의 테스트 예측 저장
    fold_test_pred = m.predict(X_test)
    fold_test_preds.append(fold_test_pred)

    # (참고용) 개별 모델의 테스트 성능
    t_mse = mean_squared_error(y_test, fold_test_pred)
    t_pcc = pearsonr(y_test, fold_test_pred)[0] if len(y_test) > 1 else np.nan
    test_mse_list.append(t_mse)
    test_pcc_list.append(t_pcc)

# --------------------------
# 앙상블(평균) 예측 및 평가
# --------------------------
y_test_pred_ens = np.mean(fold_test_preds, axis=0)

oof_mse  = mean_squared_error(y_train_all, oof_pred)
oof_pcc  = pearsonr(y_train_all, oof_pred)[0] if len(y_train_all) > 1 else np.nan
ens_mse  = mean_squared_error(y_test, y_test_pred_ens)
ens_pcc  = pearsonr(y_test, y_test_pred_ens)[0] if len(y_test) > 1 else np.nan

print("
=== CV(5) Diagnostics ===")
print(f"Fold Val MSE mean±std: {np.mean(val_mse_list):.4f} ± {np.std(val_mse_list):.4f}")
print(f"Fold Val PCC mean±std: {np.mean(val_pcc_list):.4f} ± {np.std(val_pcc_list):.4f}")
print(f"OOF MSE: {oof_mse:.4f}, OOF PCC: {oof_pcc:.4f}")

print("
=== Individual Fold Models on Test (reference) ===")
print(f"Test MSE mean±std: {np.mean(test_mse_list):.4f} ± {np.std(test_mse_list):.4f}")
print(f"Test PCC mean±std: {np.mean(test_pcc_list):.4f} ± {np.std(test_pcc_list):.4f}")

print("
=== Ensemble (Mean of 5 folds) on Test ===")
print(f"Ensemble Test MSE: {ens_mse:.4f}")
print(f"Ensemble Test PCC: {ens_pcc:.4f}")

# --- 예측/라벨/메트릭 저장
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
}
with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# --------------------------
# 테스트 산점도 (앙상블 예측)
# --------------------------
y_all_min = float(min(y_train_all.min(), y_test.min()))
y_all_max = float(max(y_train_all.max(), y_test.max()))
pad = 0.05 * (y_all_max - y_all_min if y_all_max > y_all_min else 1.0)
xmin, xmax = y_all_min - pad, y_all_max + pad
ymin, ymax = xmin, xmax

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred_ens, alpha=0.7)
plt.plot([xmin, xmax], [ymin, ymax], linestyle='--', color ='red')
plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
plt.xlabel("True (Test)")
plt.ylabel("Predicted (Test)")
plt.title(f'EGFR_T790M_L858R – LGBM+Optuna – 5Fold Ensemble
Test MSE={ens_mse:.3f}, PCC={ens_pcc:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(ckpt_dir, "scatter_test_ensemble.png"), dpi=200)
plt.show()

# --------------------------
# (선택) 저장된 텍스트 ckpt 재로드 후 평균 예측하는 간단한 유틸
# --------------------------
class LGBMTextEnsemble:
    def __init__(self, model_files):
        self.boosters = [lgb.Booster(model_file=f) for f in model_files]
    def predict(self, X):
        preds = []
        for b in self.boosters:
            # best_iteration 정보가 모델 파일에 있으면 LightGBM이 내부적으로 사용
            preds.append(b.predict(X))
        return np.mean(preds, axis=0)

# 재로드 테스트 (원하면 주석 해제)
# model_files = [os.path.join(ckpt_dir, f"model_fold{i}_best.txt") for i in range(1, n_splits+1)]
# ens_loader = LGBMTextEnsemble(model_files)
# y_pred_reload = ens_loader.predict(X_test)
# print("Reloaded ensemble MSE:", mean_squared_error(y_test, y_pred_reload))
