# Quick Start for macOS 🍎

## ✅ Setup Complete!

Your environment is ready to go. Here's what's installed:
- Python 3.14 ✓
- NumPy, Pandas, Scikit-learn ✓
- XGBoost, LightGBM ✓
- Matplotlib, Seaborn ✓
- Virtual environment (`venv/`) ✓

## 🚀 Run Training (5 minutes)

```bash
./run_quick_start.sh
```

This will:
1. Train XGBoost and LightGBM models
2. Create ensemble predictions
3. Save submission files in `submissions/`

## 📊 Results

After training, you'll see:
- **Model Performance**: Cross-validation RMSE scores
- **Submission Files**:
  - `submission_quick_ensemble.csv` - Weighted ensemble (recommended)
  - `submission_quick_XGBoost.csv` - Best individual model

## 🎯 Submit to Kaggle

1. Go to the competition page
2. Click "Submit Predictions"
3. Upload `submissions/submission_quick_ensemble.csv`
4. Check your score!

## ⚙️ Customize Training

Train specific models:
```bash
# Activate virtual environment
source venv/bin/activate

# Train only XGBoost
python quick_train.py --models xgboost --n_folds 5

# Train both XGBoost and LightGBM with 10-fold CV
python quick_train.py --models xgboost lightgbm --n_folds 10

# Use standard scaler instead of robust
python quick_train.py --models xgboost lightgbm --scaler standard

# No PCA dimensionality reduction
python quick_train.py --models xgboost --pca 0
```

## 📈 Your Results

From the quick start run:
- **XGBoost**: CV RMSE = 0.803344 (±0.091)
- **LightGBM**: CV RMSE = 0.815443 (±0.087)
- **Ensemble**: Weighted average of both models

## 🔧 Optional: Install CatBoost

CatBoost requires additional build tools:
```bash
# Install Ninja build system
brew install ninja cmake

# Install CatBoost
source venv/bin/activate
pip install catboost

# Run with CatBoost included
python quick_train.py --models xgboost lightgbm catboost
```

## 📚 Full Training Pipeline

For comprehensive training with 20+ models:
```bash
source venv/bin/activate
python train_and_predict.py
```

This trains:
- 10 traditional ML models (Ridge, Lasso, RandomForest, etc.)
- 3 gradient boosting models (XGBoost, LightGBM, CatBoost*)
- Multiple ensemble combinations

*Requires CatBoost installation

## 💡 Tips for Better Scores

1. **Use Ensemble**: Ensemble predictions typically perform better than individual models
2. **Increase CV Folds**: More folds = more reliable validation (but slower)
   ```bash
   python quick_train.py --models xgboost lightgbm --n_folds 10
   ```
3. **Try Different Preprocessing**: Experiment with scalers and PCA components
   ```bash
   python quick_train.py --models xgboost --scaler standard --pca 75
   ```

## 📁 Project Structure

```
sta314-kaggle-competition/
├── run_quick_start.sh          # ⭐ Quick start script (5 min)
├── quick_train.py              # Fast training with specific models
├── train_and_predict.py        # Full pipeline (30-60 min)
├── venv/                       # Virtual environment
├── data/                       # Competition data
├── submissions/                # Your prediction files
└── src/                        # Source code
    ├── preprocessing/
    ├── models/
    └── utils/
```

## 🐛 Troubleshooting

**"Command not found: ./run_quick_start.sh"**
```bash
chmod +x run_quick_start.sh
./run_quick_start.sh
```

**"Module not found" errors**
```bash
source venv/bin/activate
pip install -r requirements_simple.txt
```

**Want to start fresh?**
```bash
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
brew install libomp
pip install -r requirements_simple.txt
```

## 📖 More Information

- Full documentation: `README.md`
- Quick reference: `QUICKSTART.md`
- EDA notebook: `notebooks/01_exploratory_data_analysis.ipynb`

---

**You're all set!** Run `./run_quick_start.sh` to start training. Good luck with the competition! 🏆
