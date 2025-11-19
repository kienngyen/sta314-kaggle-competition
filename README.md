# Kaggle Competition - PhD-Level Regression Solution

A comprehensive machine learning solution implementing traditional ML, gradient boosting, and deep learning models with advanced feature engineering and ensemble methods.

## 📊 Competition Details

- **Task**: Regression (predict continuous target variable `y`)
- **Metric**: RMSE (Root Mean Squared Error)
- **Training Samples**: 302
- **Test Samples**: 8,002
- **Features**: 112 (X1-X112)

## 🏗️ Project Structure

```
sta314-kaggle-competition/
├── data/                          # Raw data files
│   ├── trainingdata.csv
│   ├── test_predictors.csv
│   └── SampleSubmission.csv
├── src/                           # Source code
│   ├── preprocessing/
│   │   └── feature_engineering.py # Feature engineering & preprocessing
│   ├── models/
│   │   ├── train_models.py        # Traditional ML & gradient boosting
│   │   └── deep_learning.py       # Deep learning architectures
│   └── utils/
│       └── helpers.py             # Utility functions
├── notebooks/                     # Jupyter notebooks for EDA
│   └── 01_exploratory_data_analysis.ipynb
├── models/                        # Saved models and summaries
│   └── model_summary.csv
├── submissions/                   # Competition submissions
├── train_and_predict.py          # Main execution script
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models and Generate Predictions

```bash
python train_and_predict.py
```

This will:
- Load and preprocess data
- Engineer features (polynomial, statistical, clustering)
- Train 20+ models with 5-fold cross-validation
- Create ensemble predictions
- Save submissions to `submissions/` folder

### 3. Explore Data (Optional)

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 🤖 Models Implemented

### Traditional Machine Learning
- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet
- **Tree-Based**: DecisionTree, RandomForest, ExtraTrees
- **Ensemble**: GradientBoosting, AdaBoost
- **SVM**: Support Vector Regression (RBF, Linear)

### Advanced Gradient Boosting
- **XGBoost**: 1000 estimators, learning_rate=0.05
- **LightGBM**: 1000 estimators, num_leaves=31
- **CatBoost**: 1000 iterations, silent mode

### Deep Learning (TensorFlow/Keras)
- **MLP**: Multi-layer perceptron with batch normalization
- **ResNet**: Residual network for tabular data
- **Wide & Deep**: Combined linear and deep components

## 🔧 Feature Engineering

### 1. Polynomial Features
- Squared terms (X²)
- Cubed terms (X³)
- Square root transforms (√X)

### 2. Statistical Aggregations
- Row-wise statistics: mean, std, min, max, median, range
- Distribution features: skewness, kurtosis
- Count features: zeros, positive, negative values

### 3. Clustering Features
- KMeans clustering (5 clusters)
- Distance to cluster centers

### 4. Preprocessing Strategies
- **Scaling**: RobustScaler, StandardScaler, MinMaxScaler
- **Dimensionality Reduction**: PCA (50, 75, or no reduction)
- **Feature Selection**: SelectKBest with f_regression

## 📈 Model Validation

- **Cross-Validation**: Stratified 5-fold CV
- **Metric**: RMSE (Root Mean Squared Error)
- **Early Stopping**: Implemented for deep learning models
- **Ensemble**: Weighted averaging based on inverse RMSE

## 📊 Results

Results are saved in `models/model_summary.csv` after training, showing:
- Cross-validation RMSE (mean ± std)
- Model rankings
- Performance comparison

Top submissions are automatically saved:
- `submission_final_best.csv` - **Recommended submission** (Ensemble Top 5)
- `submission_ensemble_top3.csv` - Ensemble of top 3 models
- `submission_ensemble_all.csv` - Ensemble of all models
- Individual model submissions for top performers

## 🔬 Advanced Techniques

### Small Dataset Handling
With only 302 training samples:
- Extensive cross-validation to assess generalization
- Regularization in all models (L1, L2, dropout)
- Feature engineering to extract maximum information
- Ensemble methods to reduce variance

### Hyperparameter Optimization
- Grid search for traditional models
- Predefined optimal configurations for gradient boosting
- Learning rate scheduling for deep learning

### Ensemble Strategy
Models are combined using weighted averaging:
```
weight_i = (1 / RMSE_i) / Σ(1 / RMSE_j)
prediction = Σ(weight_i × prediction_i)
```

## 📝 Key Features

✅ **Modular Design**: Clean separation of preprocessing, models, and utilities  
✅ **Reproducibility**: Fixed random seeds, versioned dependencies  
✅ **Best Practices**: Type hints, docstrings, PEP 8 compliance  
✅ **Comprehensive**: 20+ models from classical to deep learning  
✅ **Robust Validation**: 5-fold CV with multiple preprocessing strategies  
✅ **Production Ready**: Error handling, logging, model persistence  

## 🛠️ Customization

### Train Specific Models

```python
from src.models.train_models import ModelTrainer, TraditionalMLModels

trainer = ModelTrainer(n_folds=5)
models = TraditionalMLModels()

# Train only Random Forest
rf_model = models.get_tree_models()['RandomForest']
trainer.train_single_model(rf_model, X_train, y, X_test, 'RandomForest')
```

### Modify Preprocessing

```python
from src.preprocessing.feature_engineering import DataPreprocessor

preprocessor = DataPreprocessor(
    scaler_type='standard',  # 'robust', 'minmax', 'power'
    n_components=100,        # PCA components
    k_best=50               # Feature selection
)
```

### Adjust Deep Learning

```python
from src.models.deep_learning import DeepLearningModels

model = DeepLearningModels.create_mlp(
    input_dim=X_train.shape[1],
    hidden_layers=[512, 256, 128, 64],  # Custom architecture
    dropout_rate=0.4,
    l2_reg=0.001
)
```

## 📦 Dependencies

Core libraries:
- **Data**: numpy, pandas, scipy
- **ML**: scikit-learn, xgboost, lightgbm, catboost
- **DL**: tensorflow
- **Optimization**: optuna, hyperopt
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretation**: shap, lime

See `requirements.txt` for complete list with versions.

## 🎯 Performance Tips

1. **For Better Generalization**: Use ensemble methods (top 3-5 models)
2. **For Speed**: Train only gradient boosting models (XGBoost, LightGBM, CatBoost)
3. **For Interpretability**: Use SHAP values on tree-based models
4. **For Experimentation**: Modify preprocessing strategies and compare CV scores

## 📚 References

- **XGBoost**: Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System
- **LightGBM**: Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- **CatBoost**: Prokhorenkova et al. (2018) - CatBoost: unbiased boosting with categorical features
- **Ensemble Methods**: Dietterich (2000) - Ensemble Methods in Machine Learning

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

Created for STA314 Kaggle Competition with best practices in machine learning engineering.

---

**Note**: This is a competition solution demonstrating comprehensive ML/DL techniques. For production use, consider additional validation, monitoring, and deployment considerations.