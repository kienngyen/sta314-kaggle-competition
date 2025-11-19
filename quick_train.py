"""
Quick Training Script - Train specific models
Usage: python quick_train.py --models xgboost lightgbm catboost --epochs 50
"""
import argparse
import numpy as np
from src.utils.helpers import load_data, save_submission
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor
from src.models.train_models import TraditionalMLModels, AdvancedGradientBoosting, ModelTrainer

# Import deep learning only if needed
def get_dl_trainer():
    try:
        from src.models.deep_learning import DeepLearningTrainer
        return DeepLearningTrainer
    except ImportError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Quick model training')
    parser.add_argument('--models', nargs='+', 
                       choices=['linear', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'catboost', 'mlp', 'all'],
                       default=['xgboost', 'lightgbm', 'catboost'],
                       help='Models to train')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--pca', type=int, default=50, help='PCA components (None for no PCA)')
    parser.add_argument('--scaler', type=str, default='robust', 
                       choices=['robust', 'standard', 'minmax'],
                       help='Scaler type')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for deep learning')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)
    
    print("="*80)
    print("QUICK TRAINING SCRIPT")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"CV Folds: {args.n_folds}")
    print(f"PCA Components: {args.pca}")
    print(f"Scaler: {args.scaler}")
    
    # Load data
    print("\n[1/4] Loading data...")
    train_df, test_df, sample_submission = load_data()
    X = train_df.drop('y', axis=1)
    y = train_df['y'].values
    X_test = test_df
    test_ids = sample_submission['id'].values
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Preprocessing with built-in feature engineering
    print("[2/4] Preprocessing...")
    preprocessor = DataPreprocessor(
        scaler_type=args.scaler,
        apply_pca=args.pca > 0,
        n_components=args.pca if args.pca > 0 else None
    )
    X_train = preprocessor.fit_transform(X, y, engineer_features=True)
    X_test_final = preprocessor.transform(X_test, engineer_features=True)
    
    # Train models
    print(f"[3/4] Training models...")
    trainer = ModelTrainer(n_folds=args.n_folds)
    
    if 'all' in args.models:
        args.models = ['linear', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'catboost', 'mlp']
    
    # Traditional ML
    if any(m in args.models for m in ['linear', 'ridge', 'lasso', 'rf']):
        traditional = TraditionalMLModels()
        
        if 'linear' in args.models:
            from sklearn.linear_model import LinearRegression
            trainer.train_single_model(LinearRegression(), X_train, y, X_test_final, 'LinearRegression')
        
        if 'ridge' in args.models:
            from sklearn.linear_model import Ridge
            trainer.train_single_model(Ridge(alpha=10.0), X_train, y, X_test_final, 'Ridge')
        
        if 'lasso' in args.models:
            from sklearn.linear_model import Lasso
            trainer.train_single_model(Lasso(alpha=0.001), X_train, y, X_test_final, 'Lasso')
        
        if 'rf' in args.models:
            from sklearn.ensemble import RandomForestRegressor
            trainer.train_single_model(
                RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
                X_train, y, X_test_final, 'RandomForest'
            )
    
    # Gradient boosting
    boosting = AdvancedGradientBoosting()
    
    if 'xgboost' in args.models:
        xgb_model = boosting.get_xgboost()
        trainer.train_single_model(xgb_model, X_train, y, X_test_final, 'XGBoost')
    
    if 'lightgbm' in args.models:
        lgb_model = boosting.get_lightgbm()
        trainer.train_single_model(lgb_model, X_train, y, X_test_final, 'LightGBM')
    
    if 'catboost' in args.models:
        try:
            cat_model = boosting.get_catboost()
            trainer.train_single_model(cat_model, X_train, y, X_test_final, 'CatBoost')
        except ImportError as e:
            print(f"\n  ⚠️  Skipping CatBoost: {e}")
    
    # Deep learning
    if 'mlp' in args.models:
        DeepLearningTrainer = get_dl_trainer()
        if DeepLearningTrainer is None:
            print("\n  ⚠️  Skipping deep learning: TensorFlow not installed")
        else:
            dl_trainer = DeepLearningTrainer(n_folds=args.n_folds)
            dl_results = dl_trainer.train_all_dl_models(X_train, y, X_test_final, epochs=args.epochs)
            
            for model_name, result in dl_results.items():
                trainer.models[model_name] = None
                trainer.predictions[model_name] = result['predictions']
                trainer.cv_scores[model_name] = result['cv_score']
                trainer.cv_stds[model_name] = result['std']
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    sorted_scores = trainer.get_summary()
    
    # Create ensemble
    ensemble_pred = trainer.create_ensemble(top_k=min(3, len(trainer.predictions)))
    
    # Save
    save_submission(test_ids, ensemble_pred, 'submission_quick_ensemble.csv')
    print("\n✓ Submission saved: submissions/submission_quick_ensemble.csv")
    
    # Save best individual
    best_model_name = sorted_scores[0][0]
    if best_model_name in trainer.predictions:
        save_submission(test_ids, trainer.predictions[best_model_name], 
                       f'submission_quick_{best_model_name}.csv')
        print(f"✓ Best model saved: submissions/submission_quick_{best_model_name}.csv")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
