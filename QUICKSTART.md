# Quick Reference Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python train_and_predict.py
```
This trains 20+ models and creates ensemble submissions.

### 3. Quick Training (Specific Models)
```bash
# Train only gradient boosting models
python quick_train.py --models xgboost lightgbm catboost

# Train all models
python quick_train.py --models all

# Train with custom settings
python quick_train.py --models xgboost lightgbm --n_folds 10 --pca 75
```

### 4. Explore Data
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 📁 Project Structure

```
sta314-kaggle-competition/
├── train_and_predict.py      # Main pipeline (trains all models)
├── quick_train.py             # Quick training script (specific models)
├── requirements.txt           # Python dependencies
│
├── data/                      # Competition data
│   ├── trainingdata.csv       # Training set (302 samples, 112 features + target)
│   ├── test_predictors.csv    # Test set (8002 samples, 112 features)
│   └── SampleSubmission.csv   # Submission format
│
├── src/                       # Source code
│   ├── utils/
│   │   └── helpers.py         # Utility functions (load_data, rmse, save_submission)
│   ├── preprocessing/
│   │   └── feature_engineering.py  # FeatureEngineer, DataPreprocessor
│   └── models/
│       ├── train_models.py    # Traditional ML + Gradient Boosting
│       └── deep_learning.py   # Neural networks (MLP, ResNet, Wide&Deep)
│
├── notebooks/                 # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
│
├── models/                    # Saved models and summaries
│   └── model_summary.csv      # Performance comparison (created after training)
│
└── submissions/               # Competition submissions (created after training)
    ├── submission_final_best.csv        # ⭐ RECOMMENDED
    ├── submission_ensemble_top3.csv
    ├── submission_ensemble_top5.csv
    └── submission_[model_name].csv      # Individual models
```

## 🤖 Available Models

### Traditional ML (10 models)
- LinearRegression, Ridge, Lasso, ElasticNet
- DecisionTree, RandomForest, ExtraTrees
- GradientBoosting, AdaBoost
- SVR (RBF, Linear)

### Gradient Boosting (3 models)
- XGBoost (1000 estimators, lr=0.05)
- LightGBM (1000 estimators, leaves=31)
- CatBoost (1000 iterations)

### Deep Learning (3 models)
- MLP (Multi-Layer Perceptron)
- ResNet (Residual Network)
- Wide & Deep

## 📊 Expected Output

After running `train_and_predict.py`:

```
MODEL PERFORMANCE SUMMARY (sorted by CV RMSE)
Model                  CV_RMSE_Mean  CV_RMSE_Std
XGBoost                0.123456      0.012345
LightGBM               0.125678      0.013456
CatBoost               0.127890      0.014567
...
```

Submissions saved:
- ✅ `submission_final_best.csv` - Ensemble of top 5 models
- ✅ `submission_XGBoost.csv` - Best individual model
- ✅ `submission_ensemble_top3.csv` - Ensemble of top 3
- ✅ Individual submissions for all models

## ⚙️ Customization Examples

### Train Only XGBoost with 10-Fold CV
```bash
python quick_train.py --models xgboost --n_folds 10
```

### Use StandardScaler Instead of RobustScaler
```bash
python quick_train.py --models xgboost lightgbm --scaler standard
```

### No PCA Dimensionality Reduction
```bash
python quick_train.py --models xgboost --pca 0
```

### Modify in Code
```python
from src.preprocessing.feature_engineering import DataPreprocessor

# Custom preprocessing
preprocessor = DataPreprocessor(
    scaler_type='minmax',    # 'robust', 'standard', 'minmax', 'power'
    n_components=100,        # PCA components (None = no PCA)
    k_best=50               # Feature selection (None = no selection)
)
```

## 🔧 Preprocessing Pipeline

1. **Feature Engineering** (FeatureEngineer)
   - Polynomial features: X², X³, √X
   - Statistical features: mean, std, min, max, median, range, skew, kurtosis
   - Clustering features: KMeans (5 clusters) + distances

2. **Preprocessing** (DataPreprocessor)
   - Scaling: RobustScaler (default), StandardScaler, MinMaxScaler
   - PCA: 50 components (default)
   - Feature Selection: SelectKBest (optional)

## 📈 Performance Tips

### For Best Score
- Use ensemble methods: `submission_final_best.csv`
- Train all models: `python train_and_predict.py`
- Increase CV folds: `--n_folds 10`

### For Speed
- Train only boosting: `python quick_train.py --models xgboost lightgbm catboost`
- Reduce PCA: `--pca 30`

### For Experimentation
- Try different scalers: `--scaler standard` or `--scaler minmax`
- Modify PCA: `--pca 75` or `--pca 0` (no PCA)
- Different CV: `--n_folds 7`

## 🐛 Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Memory Issues
```bash
# Use quick_train with fewer models
python quick_train.py --models xgboost lightgbm
```

### Slow Training
- Reduce epochs: `--epochs 50`
- Use fewer models
- Reduce PCA components: `--pca 30`

## 📚 Key Files

### Main Execution
- `train_and_predict.py` - Full pipeline with all models
- `quick_train.py` - Fast training with specific models

### Core Modules
- `src/utils/helpers.py` - load_data(), rmse(), save_submission()
- `src/preprocessing/feature_engineering.py` - FeatureEngineer, DataPreprocessor
- `src/models/train_models.py` - TraditionalMLModels, AdvancedGradientBoosting, ModelTrainer
- `src/models/deep_learning.py` - DeepLearningModels, DeepLearningTrainer

### Analysis
- `notebooks/01_exploratory_data_analysis.ipynb` - EDA with visualizations

## 🎯 Recommended Workflow

1. **Explore Data** (5 min)
   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

2. **Quick Baseline** (5 min)
   ```bash
   python quick_train.py --models xgboost lightgbm catboost
   ```

3. **Full Training** (30-60 min)
   ```bash
   python train_and_predict.py
   ```

4. **Submit** (1 min)
   - Upload `submissions/submission_final_best.csv`
   - Check `models/model_summary.csv` for performance

## 📊 Competition Metric

**RMSE (Root Mean Squared Error)**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Lower is better. Ensemble methods typically achieve the best RMSE.

## 💡 Tips for Winning

1. ✅ Use ensemble of top 3-5 models
2. ✅ Extensive cross-validation (5-10 folds)
3. ✅ Feature engineering (polynomial, statistical)
4. ✅ Proper preprocessing (scaling, PCA)
5. ✅ Gradient boosting models (XGBoost, LightGBM, CatBoost)
6. ✅ Regularization (L1, L2, dropout)
7. ✅ Multiple preprocessing strategies

---

**Need Help?** Check README.md for detailed documentation.
