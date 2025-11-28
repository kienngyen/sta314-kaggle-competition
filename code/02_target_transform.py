import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBRegressor
from scipy.stats import yeojohnson
from sklearn.feature_selection import SelectFromModel

# Import from src
import sys
sys.path.insert(0, os.path.abspath('.'))
from src.utils.helpers import load_data, save_submission, rmse
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor

RANDOM_STATE = 42

class TargetTransformer:
    """
    A helper class to change the shape of 'y' (the target).
    Think of this like reshaping a piece of clay to make it easier to work with.
    """
    def __init__(self, method='yeo-johnson'):
        self.method = method
        self.transformer = None
        self.lambda_ = None
    
    def fit_transform(self, y):
        # If we chose Yeo-Johnson, we try to make the data look like Normal Distribution.
        if self.method == 'yeo-johnson':
            y_transformed, self.lambda_ = yeojohnson(y)
            return y_transformed
        # If we chose Quantile, we force the data to be a Bell Curve by rank.
        elif self.method == 'quantile':
            self.transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            return self.transformer.fit_transform(y.reshape(-1, 1)).ravel()
        # Otherwise, do nothing.
        else:
            return y
    
    def inverse_transform(self, y_transformed):
        # After the model predicts, we have to turn the answer back to the original shape.
        if self.method == 'yeo-johnson' and self.lambda_ is not None:
            if self.lambda_ == 0:
                return np.exp(y_transformed) - 1
            else:
                return np.power(self.lambda_ * y_transformed + 1, 1/float(self.lambda_)) - 1
        elif self.transformer is not None:
            return self.transformer.inverse_transform(y_transformed.reshape(-1, 1)).ravel()
        else:
            return y_transformed

def create_multi_level_stacking(X_train, y_train):
    """
    Creates a 'Team' of models (Stacking).
    Level 0: Base Models (ExtraTrees, XGBoost, GradientBoosting, Ridge)
    Level 1: Meta Model (Ridge Regression) who combines their answers.
    """
    print("Creating the Model Team (Stacking Ensemble)...")
    
    # Level 0: Base Models (4 models only)
    level0_models = [
        ('et', ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_split=3, 
                                   max_features=0.7, random_state=42, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=5, 
                             subsample=0.8, colsample_bytree=0.7, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.03,
                                        max_depth=5, random_state=42)),
        ('ridge', Ridge(alpha=10.0))
    ]
    
    # Level 1: Meta Model
    meta_model = Ridge(alpha=5.0)
    
    # Create the Stacking Regressor
    stacking = StackingRegressor(
        estimators=level0_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    # Train the whole team
    stacking.fit(X_train, y_train)
    
    # Check how good the team is using Cross-Validation
    # This simulates how the model will perform on new data.
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    print("Checking performance (Cross-Validation):")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # We have to create a temporary team for each fold to be fair
        temp_stack = StackingRegressor(
            estimators=level0_models,
            final_estimator=Ridge(alpha=5.0),
            cv=3
        )
        temp_stack.fit(X_tr, y_tr)
        val_pred = temp_stack.predict(X_val)
        fold_rmse = rmse(y_val, val_pred)
        cv_scores.append(fold_rmse)
        print(f"  Fold {fold} Score: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    print(f"  Average Score: {mean_rmse:.6f} (+/- {std_rmse:.6f})")
    
    return stacking, mean_rmse, std_rmse

def main():
    print("="*80)
    print("Target Transformation & Stacking")
    print("="*80)
    
    # Step 1: Get the Data
    print("[Step 1] Loading data...")
    train_df, test_df, sample_submission = load_data()
    
    X = train_df.drop('y', axis=1).values
    y = train_df['y'].values
    
    if 'id' in test_df.columns:
        test_ids = np.array(test_df['id'].values)
        X_test = test_df.drop('id', axis=1).values
    else:
        test_ids = np.array(sample_submission['id'].values)
        X_test = test_df.values
    
    # Step 2: Create New Features
    print("[Step 2] Engineering Features")
    feature_engineer = FeatureEngineer()
    X_eng = feature_engineer.fit_transform(X)
    X_test_eng = feature_engineer.transform(X_test)
    
    # Step 3: Preprocessing
    # We use RobustScaler to handle outliers (extreme values).
    print("[Step 3] Preprocessing")
    preprocessor = DataPreprocessor(scaler_type='robust', n_components=None)
    X_processed = preprocessor.fit_transform(pd.DataFrame(X_eng), pd.Series(y))
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test_eng))
    
    # Step 4: Pick the Best Features
    print("[Step 4] Feature Selection")
    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold='median'
    )
    X_selected = selector.fit_transform(X_processed, np.array(y))
    X_test_selected = selector.transform(X_test_processed)
    print(f"  We kept {X_selected.shape[1]} features.")
    
    # Step 5: Train the Models
    print("[Step 5] Running the Experiments...")
    results = {}
    
    # Experiment 1: Standard (Do nothing to y)
    print("--- Experiment 1: Standard ---")
    stack1, rmse1, std1 = create_multi_level_stacking(X_selected, y)
    results['standard'] = {'model': stack1, 'rmse': rmse1, 'y_pred': stack1.predict(X_test_selected)}
    
    # Experiment 2: Yeo-Johnson
    print("--- Experiment 2: Yeo-Johnson ---")
    target_tf = TargetTransformer(method='yeo-johnson')
    y_transformed = target_tf.fit_transform(y)
    
    # Train on the transformed target
    stack2, rmse2, std2 = create_multi_level_stacking(X_selected, y_transformed)
    
    # Predict and then flip the answer back to normal
    y_pred_tf = stack2.predict(X_test_selected)
    y_pred_original = target_tf.inverse_transform(y_pred_tf)
    results['yeo_johnson'] = {'model': stack2, 'rmse': rmse2, 'y_pred': y_pred_original}
    
    # Experiment 3: Quantile
    print("--- Experiment 3: Quantile ---")
    target_tf3 = TargetTransformer(method='quantile')
    y_transformed3 = target_tf3.fit_transform(y)
    
    stack3, rmse3, std3 = create_multi_level_stacking(X_selected, y_transformed3)
    
    y_pred_tf3 = stack3.predict(X_test_selected)
    y_pred_original3 = target_tf3.inverse_transform(y_pred_tf3)
    results['quantile'] = {'model': stack3, 'rmse': rmse3, 'y_pred': y_pred_original3}
    
    # Experiment 4: Blend
    print("--- Experiment 4: Blending ---")
    # We trust the better models more (Inverse RMSE weighting)
    weights = [1/r['rmse'] for r in results.values()]
    weights = np.array(weights) / sum(weights)
    
    blend_pred = np.average([r['y_pred'] for r in results.values()], axis=0, weights=weights)
    blend_rmse = np.average([r['rmse'] for r in results.values()], weights=weights)
    results['blend'] = {'y_pred': blend_pred, 'rmse': blend_rmse}
    
    # Save the answers
    print("[Step 6] Saving Results...")
    save_submission(results['standard']['y_pred'], test_ids, 'submission_advanced_standard.csv')
    save_submission(results['yeo_johnson']['y_pred'], test_ids, 'submission_advanced_yeo_johnson.csv')
    save_submission(results['blend']['y_pred'], test_ids, 'submission_advanced_blend.csv')
    
    print("Summary:")
    print(f"  Standard RMSE:    {results['standard']['rmse']:.6f}")
    print(f"  Yeo-Johnson RMSE: {results['yeo_johnson']['rmse']:.6f}")
    print(f"  Quantile RMSE:    {results['quantile']['rmse']:.6f}")
    print(f"  Blend RMSE:       {results['blend']['rmse']:.6f}")

if __name__ == "__main__":
    np.random.seed(42)
    main()
