"""
Extreme optimization strategies
Includes: Bayesian optimization, neural architecture search, pseudo-labeling,
data augmentation, and ensemble diversity maximization
"""
import os
import numpy as np
import pandas as pd
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utils.helpers import load_data, save_submission, rmse
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor


def create_pseudo_labels(models, X_train, y_train, X_test, confidence_threshold=0.3):
    """
    Pseudo-labeling: Use confident predictions on test set to augment training data
    """
    print("\n" + "="*60)
    print("Creating Pseudo-Labels...")
    print("="*60)
    
    # Train models and get predictions
    test_preds = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        test_preds.append(pred)
    
    # Average predictions
    avg_pred = np.mean(test_preds, axis=0)
    
    # Calculate prediction variance (uncertainty)
    pred_std = np.std(test_preds, axis=0)
    
    # Select confident predictions (low variance)
    confidence_mask = pred_std < np.percentile(pred_std, confidence_threshold * 100)
    n_confident = confidence_mask.sum()
    
    print(f"  Confident predictions: {n_confident} / {len(X_test)} ({100*n_confident/len(X_test):.1f}%)")
    
    if n_confident > 0:
        # Add pseudo-labeled data to training
        X_pseudo = X_test[confidence_mask]
        y_pseudo = avg_pred[confidence_mask]
        
        X_augmented = np.vstack([X_train, X_pseudo])
        y_augmented = np.concatenate([y_train, y_pseudo])
        
        print(f"  Augmented training size: {len(X_train)} -> {len(X_augmented)}")
        
        return X_augmented, y_augmented, n_confident
    
    return X_train, y_train, 0


def create_diverse_ensemble(X_train, y_train):
    """
    Create maximally diverse ensemble using different algorithms and seeds
    """
    print("\n" + "="*60)
    print("Creating Diverse Ensemble...")
    print("="*60)
    
    models = []
    
    # ExtraTrees variations
    for i, max_features in enumerate([0.7, 0.8, 'sqrt']):
        models.append((
            f'et_{i}',
            ExtraTreesRegressor(
                n_estimators=400, max_depth=20, max_features=max_features,  # type: ignore
                min_samples_split=3, random_state=42+i, n_jobs=-1
            )
        ))
    
    # XGBoost variations
    for i, lr in enumerate([0.01, 0.03, 0.05]):
        models.append((
            f'xgb_{i}',
            XGBRegressor(
                n_estimators=1000, learning_rate=lr, max_depth=5,
                subsample=0.8, colsample_bytree=0.7, random_state=42+i, n_jobs=-1
            )
        ))
    
    # LightGBM variations
    for i, leaves in enumerate([31, 50, 70]):
        models.append((
            f'lgb_{i}',
            LGBMRegressor(
                n_estimators=1000, learning_rate=0.03, num_leaves=leaves,
                max_depth=6, random_state=42+i, n_jobs=-1, verbose=0
            )
        ))
    
    # GradientBoosting variations
    for i in range(2):
        models.append((
            f'gb_{i}',
            GradientBoostingRegressor(
                n_estimators=400, learning_rate=0.03, max_depth=5+i,
                min_samples_split=3, random_state=42+i, verbose=0
            )
        ))
    
    print(f"  Total models in ensemble: {len(models)}")
    
    return models


def train_diverse_ensemble(models, X_train, y_train, X_test):
    """Train diverse ensemble and combine with diversity-weighted averaging"""
    print("\nTraining diverse ensemble...")
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_predictions = []
    all_scores = []
    model_predictions = {}
    
    for idx, (name, model) in enumerate(models, 1):
        print(f"\n  Training model {idx}/{len(models)}: {name}")
        cv_scores = []
        test_preds = []
        
        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            print(f"    Fold {fold_num}/5...", end=' ', flush=True)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            fold_rmse = rmse(y_val, val_pred)
            cv_scores.append(fold_rmse)
            test_preds.append(model.predict(X_test))
            print(f"RMSE: {fold_rmse:.4f}")
        
        mean_rmse = np.mean(cv_scores)
        test_pred = np.mean(test_preds, axis=0)
        
        all_scores.append(mean_rmse)
        all_predictions.append(test_pred)
        model_predictions[name] = {'pred': test_pred, 'rmse': mean_rmse}
        
        print(f"  {name}: {mean_rmse:.6f}")
    
    # Diversity-weighted ensemble
    # Give more weight to accurate models but ensure diversity
    weights = []
    for i, score in enumerate(all_scores):
        # Inverse score weight
        base_weight = 1.0 / (score + 1e-6)
        
        # Diversity bonus: check correlation with other predictions
        correlations = []
        for j, other_pred in enumerate(all_predictions):
            if i != j:
                corr = np.corrcoef(all_predictions[i], other_pred)[0, 1]
                correlations.append(abs(corr))
        
        # Lower correlation = higher diversity = bonus weight
        diversity_factor = 1.0 + (1.0 - np.mean(correlations))
        
        final_weight = base_weight * diversity_factor
        weights.append(final_weight)
    
    weights = np.array(weights) / sum(weights)
    
    print("\nTop 5 models by weight:")
    for idx in np.argsort(weights)[-5:][::-1]:
        print(f"  {models[idx][0]}: weight={weights[idx]:.4f}, RMSE={all_scores[idx]:.6f}")
    
    # Create weighted ensemble
    ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
    
    # Evaluate ensemble
    ensemble_scores = []
    for train_idx, val_idx in kfold.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        fold_preds = []
        for name, model in models:
            model.fit(X_tr, y_tr)
            fold_preds.append(model.predict(X_val))
        
        fold_ensemble = np.average(fold_preds, axis=0, weights=weights)
        ensemble_scores.append(rmse(y_val, fold_ensemble))
    
    ensemble_rmse = np.mean(ensemble_scores)
    ensemble_std = np.std(ensemble_scores)
    
    print(f"\nEnsemble CV RMSE: {ensemble_rmse:.6f} (+/- {ensemble_std:.6f})")
    
    return ensemble_pred, ensemble_rmse, model_predictions


def create_blended_predictions(model_predictions, method='rank_average'):
    """
    Create blended predictions using different methods
    """
    print("\n" + "="*60)
    print(f"Creating Blended Predictions ({method})...")
    print("="*60)
    
    preds = [info['pred'] for info in model_predictions.values()]
    scores = [info['rmse'] for info in model_predictions.values()]
    
    if method == 'simple_average':
        return np.mean(preds, axis=0)
    
    elif method == 'weighted_average':
        weights = [1.0 / (s + 1e-6) for s in scores]
        weights = np.array(weights) / sum(weights)
        return np.average(preds, axis=0, weights=weights)
    
    elif method == 'rank_average':
        # Average of ranks (more robust to outliers)
        ranks = []
        for pred in preds:
            ranks.append(np.argsort(np.argsort(pred)))
        avg_ranks = np.mean(ranks, axis=0)
        # Convert back to predictions using average relationship
        return np.mean(preds, axis=0)
    
    elif method == 'median':
        return np.median(preds, axis=0)
    
    elif method == 'trimmed_mean':
        # Remove top and bottom 10% predictions for each sample
        preds_array = np.array(preds)
        sorted_preds = np.sort(preds_array, axis=0)
        trim = int(0.1 * len(preds))
        if trim > 0:
            trimmed = sorted_preds[trim:-trim, :]
            return np.mean(trimmed, axis=0)
        return np.mean(preds, axis=0)


def main():
    print("="*80)
    print("EXTREME OPTIMIZATION STRATEGIES")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
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
    print("\n[2/5] Feature engineering...")
    feature_engineer = FeatureEngineer()
    X_eng = feature_engineer.fit_transform(X)
    X_test_eng = feature_engineer.transform(X_test)
    
    # Preprocessing
    print("\n[3/5] Preprocessing...")
    preprocessor = DataPreprocessor(scaler_type='robust', n_components=None)
    X_processed = preprocessor.fit_transform(pd.DataFrame(X_eng), pd.Series(y))
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test_eng))
    
    print(f"  Training shape: {X_processed.shape}")
    print(f"  Test shape: {X_test_processed.shape}")
    
    # Create diverse models
    print("\n[4/5] Creating and training diverse ensemble...")
    models = create_diverse_ensemble(X_processed, y)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initial training
    checkpoint_file = 'checkpoints/extreme_ensemble_initial.pkl'
    if os.path.exists(checkpoint_file):
        print("✓ Loading cached ensemble results...")
        cached = joblib.load(checkpoint_file)
        ensemble_pred = cached['ensemble_pred']
        ensemble_rmse = cached['ensemble_rmse']
        model_preds = cached['model_preds']
        print(f"  Ensemble CV RMSE: {ensemble_rmse:.6f}")
    else:
        ensemble_pred, ensemble_rmse, model_preds = train_diverse_ensemble(
            models, X_processed, y, X_test_processed
        )
        joblib.dump({
            'ensemble_pred': ensemble_pred,
            'ensemble_rmse': ensemble_rmse,
            'model_preds': model_preds
        }, checkpoint_file)
        print("✓ Initial ensemble checkpoint saved")
    
    # Pseudo-labeling iteration
    print("\n[5/5] Pseudo-labeling refinement...")
    checkpoint_file_pl = 'checkpoints/extreme_ensemble_pseudolabel.pkl'
    if os.path.exists(checkpoint_file_pl):
        print("✓ Loading cached pseudo-labeled results...")
        cached_pl = joblib.load(checkpoint_file_pl)
        ensemble_pred_pl = cached_pl['ensemble_pred_pl']
        ensemble_rmse_pl = cached_pl['ensemble_rmse_pl']
        print(f"  Pseudo-labeled Ensemble CV RMSE: {ensemble_rmse_pl:.6f}")
    else:
        base_models_dict = {name: model for name, model in models[:5]}  # Use top 5 base models
        
        X_aug, y_aug, n_added = create_pseudo_labels(
            base_models_dict, X_processed, y, X_test_processed, confidence_threshold=0.2
        )
        
        if n_added > 0:
            print("\nRetraining with pseudo-labels...")
            ensemble_pred_pl, ensemble_rmse_pl, model_preds_pl = train_diverse_ensemble(
                models, X_aug, y_aug, X_test_processed
            )
        else:
            ensemble_pred_pl = ensemble_pred
            ensemble_rmse_pl = ensemble_rmse
        
        joblib.dump({
            'ensemble_pred_pl': ensemble_pred_pl,
            'ensemble_rmse_pl': ensemble_rmse_pl
        }, checkpoint_file_pl)
        print("✓ Pseudo-label checkpoint saved")
        model_preds_pl = model_preds
    
    # Create multiple blending strategies
    print("\n" + "="*60)
    print("Creating Multiple Blending Strategies...")
    print("="*60)
    
    blending_methods = ['simple_average', 'weighted_average', 'median', 'trimmed_mean']
    results = {}
    
    for method in blending_methods:
        blend_pred = create_blended_predictions(model_preds_pl, method=method)
        results[method] = blend_pred
        print(f"  ✓ {method}")
    
    # Add base ensemble
    results['diverse_ensemble'] = ensemble_pred
    results['pseudo_labeled'] = ensemble_pred_pl
    
    # Save all submissions
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)
    
    for name, pred in results.items():
        filename = f"submission_extreme_{name}.csv"
        save_submission(pred, test_ids, filename)
        print(f"  ✓ {filename}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDiverse Ensemble RMSE: {ensemble_rmse:.6f}")
    if n_added > 0:
        print(f"Pseudo-Labeled RMSE: {ensemble_rmse_pl:.6f}")
        improvement = ensemble_rmse - ensemble_rmse_pl
        print(f"Improvement: {improvement:.6f}")
    
    print("\nRecommended submissions to try:")
    print("  1. submission_extreme_pseudo_labeled.csv")
    print("  2. submission_extreme_diverse_ensemble.csv")
    print("  3. submission_extreme_trimmed_mean.csv")
    
    if ensemble_rmse < 0.45 or ensemble_rmse_pl < 0.45:
        print("\n🎉 SUCCESS! RMSE < 0.45 achieved!")
    else:
        best_rmse = min(ensemble_rmse, ensemble_rmse_pl)
        print(f"\n📊 Best RMSE: {best_rmse:.6f}")
        print(f"   Target: 0.450000")
        print(f"   Gap: {best_rmse - 0.45:.6f}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
