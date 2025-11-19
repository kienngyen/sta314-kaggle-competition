"""
Comprehensive modeling pipeline with traditional ML, gradient boosting, and deep learning
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: brew install ninja && pip install catboost")
import warnings
warnings.filterwarnings('ignore')

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.helpers import rmse, save_model, save_submission, save_metrics


class TraditionalMLModels:
    """Traditional Machine Learning Models"""
    
    @staticmethod
    def get_linear_models():
        """Get linear regression models with various regularization"""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=10.0, random_state=42),
            'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000)
        }
    
    @staticmethod
    def get_tree_models():
        """Get tree-based models"""
        return {
            'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42
            )
        }
    
    @staticmethod
    def get_svm_models():
        """Get Support Vector Machine models"""
        return {
            'SVR_rbf': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
            'SVR_linear': SVR(kernel='linear', C=1.0, epsilon=0.1)
        }


class AdvancedGradientBoosting:
    """Advanced Gradient Boosting Models"""
    
    @staticmethod
    def get_xgboost(params=None):
        """Get XGBoost regressor"""
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50
        }
        if params:
            default_params.update(params)
        
        return XGBRegressor(**{k: v for k, v in default_params.items() if k != 'early_stopping_rounds'})
    
    @staticmethod
    def get_lightgbm(params=None):
        """Get LightGBM regressor"""
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        if params:
            default_params.update(params)
        
        return LGBMRegressor(**default_params)
    
    @staticmethod
    def get_catboost(params=None):
        """Get CatBoost regressor"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: brew install ninja && pip install catboost")
        
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 50
        }
        if params:
            default_params.update(params)
        
        return CatBoostRegressor(**default_params)


class ModelTrainer:
    """Model training and evaluation"""
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.models = {}
        self.scores = {}
        self.predictions = {}
        
    def train_single_model(self, model, X_train, y_train, X_test, model_name: str):
        """
        Train a single model with cross-validation
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            model_name: Name of the model
        
        Returns:
            predictions, cv_score, std_score
        """
        print(f"\nTraining {model_name}...")
        
        # Cross-validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        test_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            model.fit(X_tr, y_tr)
            
            # Validate
            val_pred = model.predict(X_val)
            fold_score = rmse(y_val, val_pred)
            cv_scores.append(fold_score)
            
            # Test predictions
            test_pred = model.predict(X_test)
            test_preds.append(test_pred)
            
            print(f"  Fold {fold} RMSE: {fold_score:.6f}")
        
        # Average CV score
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        print(f"  Mean CV RMSE: {mean_cv_score:.6f} (+/- {std_cv_score:.6f})")
        
        # Average test predictions
        test_predictions = np.mean(test_preds, axis=0)
        
        # Train on full data for final model
        model.fit(X_train, y_train)
        
        # Store results
        self.models[model_name] = model
        self.scores[model_name] = {'mean': mean_cv_score, 'std': std_cv_score}
        self.predictions[model_name] = test_predictions
        
        return test_predictions, mean_cv_score, std_cv_score
    
    def train_all_traditional_models(self, X_train, y_train, X_test):
        """Train all traditional ML models"""
        results = {}
        
        # Linear models
        linear_models = TraditionalMLModels.get_linear_models()
        for name, model in linear_models.items():
            pred, score, std = self.train_single_model(model, X_train, y_train, X_test, name)
            results[name] = {'predictions': pred, 'cv_score': score, 'std': std}
        
        # Tree models
        tree_models = TraditionalMLModels.get_tree_models()
        for name, model in tree_models.items():
            pred, score, std = self.train_single_model(model, X_train, y_train, X_test, name)
            results[name] = {'predictions': pred, 'cv_score': score, 'std': std}
        
        # SVM models (commented out due to computation time for large datasets)
        # svm_models = TraditionalMLModels.get_svm_models()
        # for name, model in svm_models.items():
        #     pred, score, std = self.train_single_model(model, X_train, y_train, X_test, name)
        #     results[name] = {'predictions': pred, 'cv_score': score, 'std': std}
        
        return results
    
    def train_all_boosting_models(self, X_train, y_train, X_test):
        """Train all gradient boosting models"""
        results = {}
        
        # XGBoost
        xgb = AdvancedGradientBoosting.get_xgboost()
        pred, score, std = self.train_single_model(xgb, X_train, y_train, X_test, 'XGBoost')
        results['XGBoost'] = {'predictions': pred, 'cv_score': score, 'std': std}
        
        # LightGBM
        lgbm = AdvancedGradientBoosting.get_lightgbm()
        pred, score, std = self.train_single_model(lgbm, X_train, y_train, X_test, 'LightGBM')
        results['LightGBM'] = {'predictions': pred, 'cv_score': score, 'std': std}
        
        # CatBoost
        cat = AdvancedGradientBoosting.get_catboost()
        pred, score, std = self.train_single_model(cat, X_train, y_train, X_test, 'CatBoost')
        results['CatBoost'] = {'predictions': pred, 'cv_score': score, 'std': std}
        
        return results
    
    def create_ensemble(self, top_k: int = 5):
        """
        Create ensemble from top performing models using stored predictions
        
        Args:
            top_k: Number of top models to ensemble
        
        Returns:
            ensemble_predictions
        """
        print(f"\n{'='*60}")
        print("Creating Ensemble...")
        print(f"{'='*60}")
        
        # Get top k models by CV score
        sorted_models = sorted(self.scores.items(), key=lambda x: x[1]['mean'])
        top_model_names = [name for name, _ in sorted_models[:top_k]]
        
        print(f"\nTop {top_k} models for ensemble:")
        for i, name in enumerate(top_model_names, 1):
            print(f"  {i}. {name}: {self.scores[name]['mean']:.6f}")
        
        # Weighted average ensemble (inverse RMSE weighting)
        weights = []
        ensemble_preds = []
        
        for name in top_model_names:
            weight = 1.0 / (self.scores[name]['mean'] + 1e-6)  # Inverse RMSE
            weights.append(weight)
            ensemble_preds.append(self.predictions[name])
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Weighted average
        ensemble_predictions = np.average(ensemble_preds, axis=0, weights=weights)
        
        print(f"\nEnsemble weights:")
        for name, weight in zip(top_model_names, weights):
            print(f"  {name}: {weight:.4f}")
        
        return ensemble_predictions
    
    def get_summary(self):
        """Get summary of all model performances"""
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1]['mean'])
        
        print(f"\n{'Rank':<5} {'Model':<25} {'CV RMSE':<15} {'Std Dev':<10}")
        print('-' * 60)
        
        for rank, (name, score) in enumerate(sorted_scores, 1):
            print(f"{rank:<5} {name:<25} {score['mean']:<15.6f} {score['std']:<10.6f}")
        
        return sorted_scores
