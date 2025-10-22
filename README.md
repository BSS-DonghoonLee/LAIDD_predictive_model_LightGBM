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
