$ErrorActionPreference = "Stop"
Write-Host "[1/5] Downloading and validating the official Kaggle dataset..."
python src/download_dataset.py
Write-Host "[2/5] Training and evaluating models..."
python src/model_training.py
Write-Host "[3/5] Generating SHAP outputs..."
python src/explainability.py
Write-Host "[4/5] Running submission audit..."
python src/submission_check.py
Write-Host "[5/5] Complete. Launch the dashboard with:"
Write-Host "streamlit run app.py"
