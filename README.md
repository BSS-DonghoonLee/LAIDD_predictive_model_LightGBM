LightGBM + Optuna (ECFP Binary Fingerprint) — 5-Fold Ensemble

이 프로젝트는 **화합물의 SMILES를 입력으로 하여 ECFP (Morgan) Fingerprint를 계산하고**,  
**LightGBM 회귀 모델을 Optuna로 하이퍼파라미터 튜닝한 후 5-Fold Ensemble로 학습/평가**하는 Python 스크립트입니다. (2025 LAIDD Team A)

---

## Colab 실행 방법
!git clone https://github.com/BSS-DonghoonLee/LAIDD_predictive_model_LightGBM.git
%cd LAIDD_predictive_model_LightGBM
!pip install -r requirements.txt

!python LightGBM_Optuna.py \
  --csv "/content/your_target.csv" \
  --smiles "Smiles" \
  --target "pIC50" \
  --out "./checkpoints_lgbm_ensemble(your_target)" \
  --seed 42 --test-size 0.2 --splits 5 --n-trials 200 --ecfp-radius 2 --ecfp-bits 2048
