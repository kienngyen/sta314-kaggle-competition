"""
Utility functions for the Kaggle competition
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, List
import joblib
import json
from pathlib import Path


def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets
    
    Returns:
        train_df: Training data with target
        test_df: Test data without target
        sample_submission: Sample submission format
    """
    train_df = pd.read_csv(f'{data_dir}/trainingdata.csv')
    test_df = pd.read_csv(f'{data_dir}/test_predictors.csv')
    sample_submission = pd.read_csv(f'{data_dir}/SampleSubmission.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Number of features: {train_df.shape[1] - 1}")  # Excluding target
    
    return train_df, test_df, sample_submission


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (competition metric)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save_model(model, model_name: str, models_dir: str = 'models'):
    """Save trained model"""
    Path(models_dir).mkdir(exist_ok=True)
    joblib.dump(model, f'{models_dir}/{model_name}.pkl')
    print(f"Model saved: {models_dir}/{model_name}.pkl")


def load_model(model_name: str, models_dir: str = 'models'):
    """Load trained model"""
    return joblib.load(f'{models_dir}/{model_name}.pkl')


def save_submission(predictions: np.ndarray, 
                   test_ids: np.ndarray,
                   filename: str,
                   submissions_dir: str = 'submissions'):
    """
    Save predictions in competition format
    
    Args:
        predictions: Predicted values
        test_ids: Test sample IDs
        filename: Output filename
        submissions_dir: Directory to save submissions
    """
    Path(submissions_dir).mkdir(exist_ok=True)
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'y': predictions
    })
    
    filepath = f'{submissions_dir}/{filename}'
    submission_df.to_csv(filepath, index=False)
    print(f"Submission saved: {filepath}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Prediction stats - Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")


def save_metrics(metrics: Dict, filename: str, models_dir: str = 'models'):
    """Save model performance metrics"""
    Path(models_dir).mkdir(exist_ok=True)
    
    filepath = f'{models_dir}/{filename}'
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {filepath}")


def get_feature_names(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Get feature column names"""
    if exclude_cols is None:
        exclude_cols = ['y', 'id']
    
    return [col for col in df.columns if col not in exclude_cols]


class EarlyStopping:
    """Early stopping for iterative models"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
            
        return self.early_stop
