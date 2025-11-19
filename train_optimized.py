"""
Optimized training script with hyperparameter tuning and advanced techniques
to achieve RMSE < 0.45
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform

# Import custom modules
from src.utils.helpers import load_data, save_submission, rmse
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor
from src.models.train_models import ModelTrainer


def optimize_extratrees(X_train, y_train):
    """Optimize ExtraTrees hyperparameters"""
    print("\n" + "="*60)
    print("Optimizing ExtraTrees...")
    print("="*60)
    
    param_dist = {
        'n_estimators': [300, 400, 500, 600],
        'max_depth': [15, 20, 25, 30],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', 0.5, 0.7],
        'bootstrap': [True, False]
    }
    
    base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        base_model, 
        param_dist, 
        n_iter=20,
        cv=5, 
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best RMSE: {-random_search.best_score_:.6f}")
    print(f"Best params: {random_search.best_params_}")
    
    return random_search.best_estimator_


def optimize_xgboost(X_train, y_train):
    """Optimize XGBoost hyperparameters"""
    print("\n" + "="*60)
    print("Optimizing XGBoost...")
    print("="*60)
    
    param_dist = {
        'n_estimators': [500, 750, 1000, 1500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07],
        'max_depth': [4, 5, 6, 7, 8],
        'min_child_weight': [1, 2, 3, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.05, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.05, 0.1],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }
    
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        base_model, 
        param_dist, 
        n_iter=25,
        cv=5, 
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best RMSE: {-random_search.best_score_:.6f}")
    print(f"Best params: {random_search.best_params_}")
    
    return random_search.best_estimator_


def optimize_lightgbm(X_train, y_train):
    """Optimize LightGBM hyperparameters"""
    print("\n" + "="*60)
    print("Optimizing LightGBM...")
    print("="*60)
    
    param_dist = {
        'n_estimators': [500, 750, 1000, 1500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07],
        'max_depth': [4, 5, 6, 7, 8, -1],
        'num_leaves': [31, 50, 70, 100],
        'min_child_samples': [10, 20, 30, 40],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.01, 0.05, 0.1],
        'reg_lambda': [0, 0.01, 0.05, 0.1]
    }
    
    base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=0)
    
    random_search = RandomizedSearchCV(
        base_model,  # type: ignore 
        param_dist, 
        n_iter=25,
        cv=5, 
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best RMSE: {-random_search.best_score_:.6f}")
    print(f"Best params: {random_search.best_params_}")
    
    return random_search.best_estimator_


def create_stacking_ensemble(base_models, X_train, y_train):
    """Create advanced stacking ensemble with meta-learner"""
    print("\n" + "="*60)
    print("Creating Stacking Ensemble...")
    print("="*60)
    
    # Meta-learner (Ridge regression works well for stacking)
    meta_learner = Ridge(alpha=1.0)
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # Fit stacking model
    stacking_model.fit(X_train, y_train)
    
    # Cross-validate to get performance
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        temp_model = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0),
            cv=3
        )
        temp_model.fit(X_tr, y_tr)
        val_pred = temp_model.predict(X_val)
        fold_rmse = rmse(y_val, val_pred)
        cv_scores.append(fold_rmse)
        print(f"  Fold {fold} RMSE: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    print(f"  Mean CV RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")
    
    return stacking_model, mean_rmse, std_rmse


def select_top_features(X_train, y_train, feature_names, top_n=150):
    """Select top N features based on importance"""
    print("\n" + "="*60)
    print(f"Selecting top {top_n} features...")
    print("="*60)
    
    # Train a quick model to get feature importance
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {len(indices)}")
    print(f"  Cumulative importance: {importances[indices].sum():.4f}")
    
    return indices


def main():
    """Main optimized pipeline"""
    
    print("="*80)
    print("OPTIMIZED KAGGLE COMPETITION PIPELINE - Target RMSE < 0.45")
    print("="*80)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n[1/5] Loading data...")
    train_df, test_df, sample_submission = load_data()
    
    # Separate features and target
    X = train_df.drop('y', axis=1).values
    y = train_df['y'].values
    
    if 'id' in test_df.columns:
        test_ids = np.array(test_df['id'].values)
        X_test = test_df.drop('id', axis=1).values
    else:
        test_ids = np.array(sample_submission['id'].values)
        X_test = test_df.values
    
    print(f"  Training samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # ============================================================
    # 2. ENHANCED FEATURE ENGINEERING
    # ============================================================
    print("\n[2/5] Enhanced feature engineering...")
    
    feature_engineer = FeatureEngineer()
    
    # Apply feature engineering
    X_engineered = feature_engineer.fit_transform(X)
    X_test_engineered = feature_engineer.transform(X_test)
    
    print(f"  Engineered features: {X_engineered.shape[1]}")
    
    # ============================================================
    # 3. PREPROCESSING WITH FEATURE SELECTION
    # ============================================================
    print("\n[3/5] Preprocessing with feature selection...")
    
    # Try the best preprocessing from previous run
    preprocessor = DataPreprocessor(
        scaler_type='robust',
        n_components=None  # No PCA for now, we'll select features instead
    )
    
    X_processed = preprocessor.fit_transform(pd.DataFrame(X_engineered), pd.Series(y))
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test_engineered))
    
    print(f"  Processed features: {X_processed.shape[1]}")
    
    # Feature selection based on importance
    feature_indices = select_top_features(X_processed, y, None, top_n=150)
    X_train = X_processed[:, feature_indices]
    X_test_final = X_test_processed[:, feature_indices]
    
    print(f"  Final features: {X_train.shape[1]}")
    
    # ============================================================
    # 4. HYPERPARAMETER OPTIMIZATION
    # ============================================================
    print("\n[4/5] Hyperparameter optimization...")
    
    # Optimize top models
    print("\nThis may take several minutes...")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check for existing checkpoints
    if os.path.exists('checkpoints/best_et_optimized.pkl'):
        print("\n✓ Loading cached ExtraTrees model...")
        best_et = joblib.load('checkpoints/best_et_optimized.pkl')
    else:
        best_et = optimize_extratrees(X_train, y)
        joblib.dump(best_et, 'checkpoints/best_et_optimized.pkl')
        print("✓ ExtraTrees checkpoint saved")
    
    if os.path.exists('checkpoints/best_xgb_optimized.pkl'):
        print("\n✓ Loading cached XGBoost model...")
        best_xgb = joblib.load('checkpoints/best_xgb_optimized.pkl')
    else:
        best_xgb = optimize_xgboost(X_train, y)
        joblib.dump(best_xgb, 'checkpoints/best_xgb_optimized.pkl')
        print("✓ XGBoost checkpoint saved")
    
    if os.path.exists('checkpoints/best_lgb_optimized.pkl'):
        print("\n✓ Loading cached LightGBM model...")
        best_lgb = joblib.load('checkpoints/best_lgb_optimized.pkl')
    else:
        best_lgb = optimize_lightgbm(X_train, y)
        joblib.dump(best_lgb, 'checkpoints/best_lgb_optimized.pkl')
        print("✓ LightGBM checkpoint saved")
    
    # ============================================================
    # 5. CREATE STACKING ENSEMBLE
    # ============================================================
    print("\n[5/5] Creating stacking ensemble...")
    
    base_models = [
        ('extratrees', best_et),
        ('xgboost', best_xgb),
        ('lightgbm', best_lgb)
    ]
    
    stacking_model, stack_rmse, stack_std = create_stacking_ensemble(
        base_models, X_train, y
    )
    
    # ============================================================
    # GENERATE PREDICTIONS
    # ============================================================
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    # Individual model predictions
    predictions = {}
    
    print("\nEvaluating individual models...")
    for name, model in base_models:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        test_preds = []
        
        for train_idx, val_idx in kfold.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_tr, y_tr)  # type: ignore
            val_pred = model.predict(X_val)  # type: ignore
            cv_scores.append(rmse(np.array(y_val), val_pred))
            test_preds.append(model.predict(X_test_final))  # type: ignore
        
        mean_rmse = np.mean(cv_scores)
        predictions[name] = np.mean(test_preds, axis=0)
        print(f"  {name}: {mean_rmse:.6f}")
    
    # Stacking predictions
    model.fit(X_train, y)  # type: ignore  # Refit on full training data
    stacking_pred = stacking_model.predict(X_test_final)
    predictions['stacking'] = stacking_pred
    
    # ============================================================
    # SAVE SUBMISSIONS
    # ============================================================
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)
    
    for name, pred in predictions.items():
        filename = f"submission_optimized_{name}.csv"
        save_submission(pred, test_ids, filename)
        print(f"  ✓ {filename}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\nBest Stacking RMSE: {stack_rmse:.6f} (+/- {stack_std:.6f})")
    print(f"\nRecommended submission: submission_optimized_stacking.csv")
    print("\nIf RMSE is still > 0.45, consider:")
    print("  - More aggressive feature engineering")
    print("  - Trying neural networks with larger architectures")
    print("  - Ensemble with more diverse models")
    print("  - Target transformation (log, box-cox)")


if __name__ == "__main__":
    np.random.seed(42)
    main()
