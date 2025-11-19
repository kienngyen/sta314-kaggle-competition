"""
Main script for training all models and generating predictions
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.utils.helpers import load_data, save_submission, save_model
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor
from src.models.train_models import TraditionalMLModels, AdvancedGradientBoosting, ModelTrainer
from src.models.deep_learning import DeepLearningTrainer


def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("KAGGLE COMPETITION - COMPREHENSIVE MODELING PIPELINE")
    print("="*80)
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n[1/6] Loading data...")
    train_df, test_df, sample_submission = load_data()
    
    print(f"  Training data shape: {train_df.shape}")
    print(f"  Test data shape: {test_df.shape}")
    print(f"  Number of features: {train_df.shape[1] - 1}")
    
    # Separate features and target
    X = train_df.drop('y', axis=1).values
    y = train_df['y'].values
    X_test = test_df.values
    test_ids = sample_submission['id'].values
    
    # ============================================================
    # 2. FEATURE ENGINEERING
    # ============================================================
    print("\n[2/6] Feature engineering...")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer()
    
    # Apply feature engineering
    X_engineered = feature_engineer.fit_transform(X)
    X_test_engineered = feature_engineer.transform(X_test)
    
    print(f"  Original features: {X.shape[1]}")
    print(f"  Engineered features: {X_engineered.shape[1]}")
    print(f"  Features added: {X_engineered.shape[1] - X.shape[1]}")
    
    # ============================================================
    # 3. PREPROCESSING
    # ============================================================
    print("\n[3/6] Preprocessing data...")
    
    # Try different preprocessing strategies
    preprocessing_strategies = {
        'robust_pca50': {
            'scaler_type': 'robust',
            'pca_components': 50
        },
        'standard_pca75': {
            'scaler_type': 'standard',
            'pca_components': 75
        },
        'robust_no_pca': {
            'scaler_type': 'robust',
            'pca_components': None
        }
    }
    
    preprocessed_data = {}
    for strategy_name, params in preprocessing_strategies.items():
        print(f"\n  Strategy: {strategy_name}")
        preprocessor = DataPreprocessor(
            scaler_type=params['scaler_type'],
            n_components=params['pca_components']
        )
        
        X_processed = preprocessor.fit_transform(X_engineered, y)
        X_test_processed = preprocessor.transform(X_test_engineered)
        
        preprocessed_data[strategy_name] = {
            'X_train': X_processed,
            'X_test': X_test_processed,
            'preprocessor': preprocessor
        }
        
        print(f"    Final feature count: {X_processed.shape[1]}")
    
    # Use best strategy for main pipeline (robust with PCA=50)
    X_train = preprocessed_data['robust_pca50']['X_train']
    X_test_final = preprocessed_data['robust_pca50']['X_test']
    
    # ============================================================
    # 4. TRAIN TRADITIONAL ML & GRADIENT BOOSTING MODELS
    # ============================================================
    print("\n[4/6] Training traditional ML and gradient boosting models...")
    
    trainer = ModelTrainer(n_folds=5)
    
    # Train traditional ML models
    print("\n  Training Traditional ML Models...")
    traditional_models = TraditionalMLModels()
    all_traditional = traditional_models.get_all_models()
    
    for name, model in all_traditional.items():
        trainer.train_single_model(model, X_train, y, X_test_final, name)
    
    # Train gradient boosting models
    print("\n  Training Gradient Boosting Models...")
    boosting_models = AdvancedGradientBoosting()
    
    xgb_model = boosting_models.get_xgboost()
    trainer.train_single_model(xgb_model, X_train, y, X_test_final, 'XGBoost')
    
    lgb_model = boosting_models.get_lightgbm()
    trainer.train_single_model(lgb_model, X_train, y, X_test_final, 'LightGBM')
    
    cat_model = boosting_models.get_catboost()
    trainer.train_single_model(cat_model, X_train, y, X_test_final, 'CatBoost')
    
    # ============================================================
    # 5. TRAIN DEEP LEARNING MODELS
    # ============================================================
    print("\n[5/6] Training deep learning models...")
    
    dl_trainer = DeepLearningTrainer(n_folds=5)
    dl_results = dl_trainer.train_all_dl_models(
        X_train, y, X_test_final,
        epochs=100
    )
    
    # Add DL results to trainer
    for model_name, result in dl_results.items():
        trainer.models[model_name] = None  # DL models not stored
        trainer.predictions[model_name] = result['predictions']
        trainer.cv_scores[model_name] = result['cv_score']
        trainer.cv_stds[model_name] = result['std']
    
    # ============================================================
    # 6. CREATE ENSEMBLES AND SAVE PREDICTIONS
    # ============================================================
    print("\n[6/6] Creating ensembles and saving predictions...")
    
    # Get model summary
    summary_df = trainer.get_summary()
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY (sorted by CV RMSE)")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = 'models/model_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to: {summary_path}")
    
    # Create ensembles
    ensembles = {
        'top3': 3,
        'top5': 5,
        'top10': 10,
        'all': None
    }
    
    ensemble_predictions = {}
    for ensemble_name, top_k in ensembles.items():
        ensemble_pred = trainer.create_ensemble(top_k=top_k)
        ensemble_predictions[ensemble_name] = ensemble_pred
    
    # Save best individual models
    best_models = summary_df.head(5)['Model'].tolist()
    
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)
    
    # Save individual best models
    for model_name in best_models:
        if model_name in trainer.predictions:
            predictions = trainer.predictions[model_name]
            filename = f"submission_{model_name}.csv"
            save_submission(test_ids, predictions, filename)
            print(f"  ✓ {filename}")
    
    # Save ensemble predictions
    for ensemble_name, predictions in ensemble_predictions.items():
        filename = f"submission_ensemble_{ensemble_name}.csv"
        save_submission(test_ids, predictions, filename)
        print(f"  ✓ {filename}")
    
    # Create final best submission (ensemble of top 5)
    final_predictions = ensemble_predictions['top5']
    save_submission(test_ids, final_predictions, 'submission_final_best.csv')
    print(f"\n  🎯 BEST SUBMISSION: submission_final_best.csv (Ensemble Top 5)")
    
    # Print best models
    print("\n" + "="*80)
    print("TOP 5 MODELS:")
    print("="*80)
    for i, row in best_models.head().iterrows():
        model_name = summary_df.loc[i, 'Model']
        cv_score = summary_df.loc[i, 'CV_RMSE_Mean']
        cv_std = summary_df.loc[i, 'CV_RMSE_Std']
        print(f"  {i+1}. {model_name:20s} - RMSE: {cv_score:.6f} (+/- {cv_std:.6f})")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nSubmission files saved in: submissions/")
    print(f"Model summary saved in: models/model_summary.csv")
    print(f"\nRecommended submission: submission_final_best.csv")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run main pipeline
    main()
