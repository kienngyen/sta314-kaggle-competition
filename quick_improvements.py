"""
Quick improvements script - faster than full hyperparameter optimization
Implements proven techniques to reduce RMSE
"""
import os
import numpy as np
import pandas as pd
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utils.helpers import load_data, save_submission, rmse
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor


def get_optimized_models():
    """Get models with manually-tuned parameters based on common best practices"""
    
    models = {
        'ExtraTrees_v2': ExtraTreesRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost_v2': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM_v2': LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'GradientBoosting_v2': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    }
    
    return models


def train_with_cv(model, X_train, y_train, X_test, model_name, n_folds=5):
    """Train model with cross-validation"""
    print(f"\nTraining {model_name}...")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        fold_rmse = rmse(y_val, val_pred)
        cv_scores.append(fold_rmse)
        
        test_pred = model.predict(X_test)
        test_preds.append(test_pred)
        
        print(f"  Fold {fold} RMSE: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    test_predictions = np.mean(test_preds, axis=0)
    
    print(f"  Mean CV RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")
    
    return test_predictions, mean_rmse, std_rmse


def main():
    print("="*80)
    print("QUICK IMPROVEMENTS - Optimized Model Parameters")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    train_df, test_df, sample_submission = load_data()
    
    X = train_df.drop('y', axis=1).values
    y = train_df['y'].values
    
    if 'id' in test_df.columns:
        test_ids = np.array(test_df['id'].values)
        X_test = test_df.drop('id', axis=1).values
    else:
        test_ids = np.array(sample_submission['id'].values)
        X_test = test_df.values
    
    # Feature engineering
    print("\n[2/4] Feature engineering...")
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.fit_transform(X)
    X_test_engineered = feature_engineer.transform(X_test)
    
    print(f"  Engineered features: {X_engineered.shape[1]}")
    
    # Preprocessing
    print("\n[3/4] Preprocessing...")
    preprocessor = DataPreprocessor(scaler_type='robust', n_components=None)
    X_train = preprocessor.fit_transform(pd.DataFrame(X_engineered), pd.Series(y))
    X_test_final = preprocessor.transform(pd.DataFrame(X_test_engineered))
    
    print(f"  Final features: {X_train.shape[1]}")
    
    # Train optimized models
    print("\n[4/4] Training optimized models...")
    
    models = get_optimized_models()
    results = {}
    predictions = {}
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for name, model in models.items():
        checkpoint_file = f'checkpoints/quick_{name}.pkl'
        result_file = f'checkpoints/quick_{name}_results.pkl'
        
        if os.path.exists(checkpoint_file) and os.path.exists(result_file):
            print(f"\n✓ Loading cached model: {name}")
            cached = joblib.load(result_file)
            results[name] = cached['results']
            predictions[name] = cached['pred']
            print(f"  {name}: {results[name]['score']:.6f} (+/- {results[name]['std']:.6f})")
        else:
            pred, score, std = train_with_cv(model, X_train, y, X_test_final, name)
            results[name] = {'score': score, 'std': std}
            predictions[name] = pred
            joblib.dump(model, checkpoint_file)
            joblib.dump({'results': results[name], 'pred': pred}, result_file)
            print(f"✓ Checkpoint saved: {name}")
    
    # Create weighted ensemble
    print("\n" + "="*60)
    print("Creating Weighted Ensemble...")
    print("="*60)
    
    weights = []
    ensemble_preds = []
    
    for name in sorted(results.keys(), key=lambda x: results[x]['score']):
        weight = 1.0 / (results[name]['score'] + 1e-6)
        weights.append(weight)
        ensemble_preds.append(predictions[name])
        print(f"  {name}: RMSE={results[name]['score']:.6f}, Weight={weight:.4f}")
    
    weights = np.array(weights) / sum(weights)
    ensemble_pred = np.average(ensemble_preds, axis=0, weights=weights)
    
    # Evaluate ensemble with CV
    print("\nEvaluating ensemble with cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        fold_preds = []
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            fold_preds.append(model.predict(X_val))
        
        fold_ensemble = np.average(fold_preds, axis=0, weights=weights)
        fold_rmse = rmse(np.array(y_val), fold_ensemble)
        cv_scores.append(fold_rmse)
        print(f"  Fold {fold} RMSE: {fold_rmse:.6f}")
    
    ensemble_rmse = np.mean(cv_scores)
    ensemble_std = np.std(cv_scores)
    print(f"  Ensemble CV RMSE: {ensemble_rmse:.6f} (+/- {ensemble_std:.6f})")
    
    # Save submissions
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)
    
    # Save individual models
    for name, pred in predictions.items():
        filename = f"submission_quick_{name}.csv"
        save_submission(pred, test_ids, filename)
        print(f"  ✓ {filename} - CV RMSE: {results[name]['score']:.6f}")
    
    # Save ensemble
    filename = "submission_quick_ensemble.csv"
    save_submission(ensemble_pred, test_ids, filename)
    print(f"  ✓ {filename} - CV RMSE: {ensemble_rmse:.6f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBest Individual Model: {min(results.items(), key=lambda x: x[1]['score'])[0]}")
    print(f"Best Individual RMSE: {min(results.values(), key=lambda x: x['score'])['score']:.6f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.6f}")
    print(f"\nRecommended: submission_quick_ensemble.csv")
    
    if ensemble_rmse < 0.45:
        print(f"\n🎉 SUCCESS! Target RMSE < 0.45 achieved!")
    else:
        improvement_needed = ensemble_rmse - 0.45
        print(f"\n📊 Current RMSE: {ensemble_rmse:.6f}")
        print(f"   Target RMSE: 0.450000")
        print(f"   Gap: {improvement_needed:.6f}")
        print("\nFor further improvement, run train_optimized.py for full hyperparameter search")


if __name__ == "__main__":
    np.random.seed(42)
    main()
