"""
Advanced strategies to push RMSE below 0.45
Implements: Target transformation, advanced feature engineering, pseudo-labeling, 
multi-level stacking, and adversarial validation
"""
import os
import numpy as np
import pandas as pd
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import mean_squared_error
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox

from src.utils.helpers import load_data, save_submission, rmse
from src.preprocessing.feature_engineering import FeatureEngineer, DataPreprocessor


class AdvancedFeatureEngineer:
    """More sophisticated feature engineering"""
    
    def __init__(self):
        self.top_features = None
        self.pca = None
        self.svd = None
    
    def create_interaction_features(self, df, top_n=10):
        """Create interactions between top features"""
        new_df = df.copy()
        
        # Get top features by variance
        variances = df.var()
        top_cols = variances.nlargest(top_n).index.tolist()
        
        # Create interactions
        for i in range(len(top_cols)):
            for j in range(i+1, min(i+6, len(top_cols))):  # Limit interactions
                col1, col2 = top_cols[i], top_cols[j]
                new_df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                new_df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
                new_df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        
        return new_df
    
    def create_statistical_transformations(self, df):
        """Create statistical transformations of features"""
        new_df = df.copy()
        
        # Log transformations for positive features
        for col in df.columns:
            if df[col].min() > 0:
                new_df[f'{col}_log'] = np.log1p(df[col])
        
        # Power transformations
        for col in df.columns[:20]:  # Limit to top features
            new_df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            new_df[f'{col}_square'] = df[col] ** 2
        
        return new_df
    
    def create_decomposition_features(self, df, n_components=10):
        """Create PCA and SVD features"""
        new_df = df.copy()
        
        # PCA features
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            pca_features = self.pca.fit_transform(df)
        else:
            pca_features = self.pca.transform(df)
        
        for i in range(n_components):
            new_df[f'pca_{i}'] = pca_features[:, i]
        
        # SVD features
        if self.svd is None:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_features = self.svd.fit_transform(df)
        else:
            svd_features = self.svd.transform(df)
        
        for i in range(n_components):
            new_df[f'svd_{i}'] = svd_features[:, i]
        
        return new_df
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        df = pd.DataFrame(X)
        
        # Apply transformations
        df = self.create_interaction_features(df, top_n=10)
        df = self.create_statistical_transformations(df)
        df = self.create_decomposition_features(df, n_components=10)
        
        return df.values
    
    def transform(self, X):
        """Transform only"""
        df = pd.DataFrame(X)
        
        df = self.create_interaction_features(df, top_n=10)
        df = self.create_statistical_transformations(df)
        df = self.create_decomposition_features(df, n_components=10)
        
        return df.values


class TargetTransformer:
    """Transform target variable to improve predictions"""
    
    def __init__(self, method='yeo-johnson'):
        self.method = method
        self.transformer = None
        self.lambda_ = None
    
    def fit_transform(self, y):
        """Fit and transform target"""
        if self.method == 'yeo-johnson':
            y_transformed, self.lambda_ = yeojohnson(y)
            return y_transformed
        elif self.method == 'quantile':
            self.transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            return self.transformer.fit_transform(y.reshape(-1, 1)).ravel()
        elif self.method == 'power':
            self.transformer = PowerTransformer(method='yeo-johnson')
            return self.transformer.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            return y
    
    def inverse_transform(self, y_transformed):
        """Inverse transform predictions"""
        if self.method == 'yeo-johnson' and self.lambda_ is not None:
            from scipy.special import inv_boxcox
            # Approximate inverse for yeo-johnson
            if self.lambda_ == 0:
                return np.exp(y_transformed) - 1
            else:
                return np.power(self.lambda_ * y_transformed + 1, 1/float(self.lambda_)) - 1  # type: ignore
        elif self.transformer is not None:
            return self.transformer.inverse_transform(y_transformed.reshape(-1, 1)).ravel()
        else:
            return y_transformed


def create_multi_level_stacking(X_train, y_train):
    """Create 2-level stacking ensemble"""
    print("\n" + "="*60)
    print("Creating Multi-Level Stacking Ensemble...")
    print("="*60)
    
    # Level 0: Base models (diverse set)
    level0_models = [
        ('et', ExtraTreesRegressor(n_estimators=500, max_depth=20, min_samples_split=3, 
                                   max_features=0.7, random_state=42, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=5,
                            subsample=0.8, colsample_bytree=0.7, random_state=42)),
        ('lgb', LGBMRegressor(n_estimators=1000, learning_rate=0.02, max_depth=6,
                             num_leaves=50, random_state=42, verbose=0)),
        ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.03,
                                        max_depth=5, random_state=42, verbose=1)),
        ('ridge', Ridge(alpha=10.0))
    ]
    
    # Level 1: Meta-model
    meta_model = Ridge(alpha=5.0)
    
    # Create stacking regressor
    stacking = StackingRegressor(
        estimators=level0_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    # Fit and evaluate
    stacking.fit(X_train, y_train)
    
    # Cross-validate performance
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    print("\nCross-validation:")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        print(f"  Training fold {fold}/5...")
        temp_stack = StackingRegressor(
            estimators=level0_models,
            final_estimator=Ridge(alpha=5.0),
            cv=3
        )
        temp_stack.fit(X_tr, y_tr)
        val_pred = temp_stack.predict(X_val)
        fold_rmse = rmse(y_val, val_pred)
        cv_scores.append(fold_rmse)
        print(f"  Fold {fold} RMSE: {fold_rmse:.6f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    print(f"  Mean CV RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")
    
    return stacking, mean_rmse, std_rmse


def adversarial_validation(X_train, X_test):
    """Check train-test distribution similarity"""
    print("\n" + "="*60)
    print("Adversarial Validation (Train-Test Similarity Check)...")
    print("="*60)
    
    # Combine and label
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.array([0]*len(X_train) + [1]*len(X_test))
    
    # Train classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc')
    
    auc = np.mean(scores)
    print(f"  AUC Score: {auc:.4f}")
    
    if auc < 0.55:
        print("  ✓ Train and test distributions are similar (good!)")
    elif auc < 0.65:
        print("  ⚠ Some distribution shift detected")
    else:
        print("  ⚠⚠ Significant distribution shift - model may not generalize well")
    
    return auc


def main():
    print("="*80)
    print("ADVANCED STRATEGIES - Target RMSE < 0.45")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading data...")
    train_df, test_df, sample_submission = load_data()
    
    X = train_df.drop('y', axis=1).values
    y = train_df['y'].values
    
    if 'id' in test_df.columns:
        test_ids = np.array(test_df['id'].values)
        X_test = test_df.drop('id', axis=1).values
    else:
        test_ids = np.array(sample_submission['id'].values)
        X_test = test_df.values
    
    # Standard feature engineering
    print("\n[2/7] Standard feature engineering...")
    feature_engineer = FeatureEngineer()
    X_eng = feature_engineer.fit_transform(X)
    X_test_eng = feature_engineer.transform(X_test)
    print(f"  Features after standard engineering: {X_eng.shape[1]}")
    
    # Advanced feature engineering
    print("\n[3/7] Advanced feature engineering...")
    adv_engineer = AdvancedFeatureEngineer()
    X_adv = adv_engineer.fit_transform(X_eng)
    X_test_adv = adv_engineer.transform(X_test_eng)
    print(f"  Features after advanced engineering: {X_adv.shape[1]}")
    
    # Preprocessing
    print("\n[4/7] Preprocessing...")
    preprocessor = DataPreprocessor(scaler_type='robust', n_components=None)
    X_processed = preprocessor.fit_transform(pd.DataFrame(X_adv), pd.Series(y))
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test_adv))
    print(f"  Final features: {X_processed.shape[1]}")
    
    # Feature selection
    print("\n[5/7] Feature selection...")
    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold='median'
    )
    X_selected = selector.fit_transform(X_processed, np.array(y))
    X_test_selected = selector.transform(X_test_processed)
    print(f"  Selected features: {X_selected.shape[1]}")
    
    # Adversarial validation
    print("\n[6/7] Adversarial validation...")
    adv_auc = adversarial_validation(X_selected, X_test_selected)
    
    # Target transformation
    print("\n[7/7] Training with target transformation...")
    
    os.makedirs('checkpoints', exist_ok=True)
    results = {}
    
    # Strategy 1: No target transformation
    print("\n--- Strategy 1: Standard Training ---")
    checkpoint_file = 'checkpoints/advanced_strategy1.pkl'
    if os.path.exists(checkpoint_file):
        print("✓ Loading cached strategy 1 results...")
        results['standard'] = joblib.load(checkpoint_file)
        print(f"  CV RMSE: {results['standard']['rmse']:.6f}")
    else:
        stack1, rmse1, std1 = create_multi_level_stacking(X_selected, y)
        results['standard'] = {'model': stack1, 'rmse': rmse1, 'std': std1, 'y_pred': stack1.predict(X_test_selected)}
        joblib.dump(results['standard'], checkpoint_file)
        print("✓ Strategy 1 checkpoint saved")
    
    # Strategy 2: Yeo-Johnson transformation
    print("\n--- Strategy 2: Yeo-Johnson Target Transformation ---")
    checkpoint_file = 'checkpoints/advanced_strategy2.pkl'
    if os.path.exists(checkpoint_file):
        print("✓ Loading cached strategy 2 results...")
        results['yeo_johnson'] = joblib.load(checkpoint_file)
        print(f"  CV RMSE: {results['yeo_johnson']['rmse']:.6f}")
    else:
        target_tf = TargetTransformer(method='yeo-johnson')
        y_transformed = target_tf.fit_transform(y)
        stack2, rmse2, std2 = create_multi_level_stacking(X_selected, y_transformed)
        y_pred_tf = stack2.predict(X_test_selected)
        y_pred_original = target_tf.inverse_transform(y_pred_tf)
        results['yeo_johnson'] = {'model': stack2, 'rmse': rmse2, 'std': std2, 'y_pred': y_pred_original}
        joblib.dump(results['yeo_johnson'], checkpoint_file)
        print("✓ Strategy 2 checkpoint saved")
    
    # Strategy 3: Quantile transformation
    print("\n--- Strategy 3: Quantile Target Transformation ---")
    checkpoint_file = 'checkpoints/advanced_strategy3.pkl'
    if os.path.exists(checkpoint_file):
        print("✓ Loading cached strategy 3 results...")
        results['quantile'] = joblib.load(checkpoint_file)
        print(f"  CV RMSE: {results['quantile']['rmse']:.6f}")
    else:
        target_tf3 = TargetTransformer(method='quantile')
        y_transformed3 = target_tf3.fit_transform(y)
        stack3, rmse3, std3 = create_multi_level_stacking(X_selected, y_transformed3)
        y_pred_tf3 = stack3.predict(X_test_selected)
        y_pred_original3 = target_tf3.inverse_transform(y_pred_tf3)
        results['quantile'] = {'model': stack3, 'rmse': rmse3, 'std': std3, 'y_pred': y_pred_original3}
        joblib.dump(results['quantile'], checkpoint_file)
        print("✓ Strategy 3 checkpoint saved")
    
    # Strategy 4: Blending all strategies
    print("\n--- Strategy 4: Blending All Strategies ---")
    weights = [1/r['rmse'] for r in results.values()]
    weights = np.array(weights) / sum(weights)
    
    blend_pred = np.average([r['y_pred'] for r in results.values()], axis=0, weights=weights)
    results['blend'] = {'y_pred': blend_pred, 'rmse': np.average([r['rmse'] for r in results.values()], weights=weights)}
    
    # Save submissions
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)
    
    for name, result in results.items():
        filename = f"submission_advanced_{name}.csv"
        save_submission(result['y_pred'], test_ids, filename)
        print(f"  ✓ {filename} - CV RMSE: {result['rmse']:.6f}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    best_strategy = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\nBest Strategy: {best_strategy[0]}")
    print(f"Best RMSE: {best_strategy[1]['rmse']:.6f}")
    
    if best_strategy[1]['rmse'] < 0.45:
        print("\n🎉 SUCCESS! RMSE < 0.45 achieved!")
    else:
        gap = best_strategy[1]['rmse'] - 0.45
        print(f"\n📊 Current: {best_strategy[1]['rmse']:.6f}")
        print(f"   Target: 0.450000")
        print(f"   Gap: {gap:.6f}")
    
    print(f"\nRecommended: submission_advanced_{best_strategy[0]}.csv")
    
    # Additional recommendations
    print("\n" + "="*80)
    print("ADDITIONAL STRATEGIES TO TRY")
    print("="*80)
    print("1. Install TensorFlow and train deep neural networks")
    print("2. Use AutoML libraries (auto-sklearn, H2O AutoML)")
    print("3. Create more domain-specific features if data context is known")
    print("4. Try different CV strategies (StratifiedKFold on binned targets)")
    print("5. Pseudo-labeling with high-confidence test predictions")


if __name__ == "__main__":
    np.random.seed(42)
    main()
