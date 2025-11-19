"""
Deep Learning Models for Tabular Data
"""
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.helpers import rmse


class DeepLearningModels:
    """Deep learning architectures for tabular data"""
    
    @staticmethod
    def create_mlp(input_dim: int, 
                   hidden_layers: list = [256, 128, 64, 32],
                   dropout_rate: float = 0.3,
                   l2_reg: float = 0.001):
        """
        Create Multi-Layer Perceptron
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            l2_reg: L2 regularization strength
        
        Returns:
            Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
            
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i}'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(1, name='output'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def create_resnet(input_dim: int,
                     n_blocks: int = 3,
                     block_size: int = 128,
                     dropout_rate: float = 0.2):
        """
        Create Residual Network for tabular data
        
        Args:
            input_dim: Number of input features
            n_blocks: Number of residual blocks
            block_size: Size of each block
            dropout_rate: Dropout rate
        
        Returns:
            Keras model
        """
        inputs = layers.Input(shape=(input_dim,))
        
        # Initial projection
        x = layers.Dense(block_size, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Residual blocks
        for i in range(n_blocks):
            residual = x
            
            x = layers.Dense(block_size, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Dense(block_size)(x)
            x = layers.BatchNormalization()(x)
            
            # Add residual connection
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def create_wide_and_deep(input_dim: int,
                            wide_dim: int = None,
                            deep_layers: list = [128, 64, 32]):
        """
        Create Wide & Deep architecture
        
        Args:
            input_dim: Number of input features
            wide_dim: Dimension for wide part (if None, uses input_dim)
            deep_layers: List of deep layer sizes
        
        Returns:
            Keras model
        """
        if wide_dim is None:
            wide_dim = input_dim
        
        inputs = layers.Input(shape=(input_dim,))
        
        # Wide part (linear)
        wide = layers.Dense(1, name='wide')(inputs)
        
        # Deep part
        deep = inputs
        for i, units in enumerate(deep_layers):
            deep = layers.Dense(units, activation='relu', name=f'deep_{i}')(deep)
            deep = layers.BatchNormalization()(deep)
            deep = layers.Dropout(0.2)(deep)
        
        deep = layers.Dense(1, name='deep_output')(deep)
        
        # Combine wide and deep
        combined = layers.Add(name='combined')([wide, deep])
        
        model = keras.Model(inputs=inputs, outputs=combined)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class DeepLearningTrainer:
    """Trainer for deep learning models"""
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.models = {}
        self.histories = {}
        
    def train_with_cv(self, 
                     model_fn,
                     X_train,
                     y_train,
                     X_test,
                     model_name: str,
                     epochs: int = 100,
                     batch_size: int = 32,
                     patience: int = 20):
        """
        Train deep learning model with cross-validation
        
        Args:
            model_fn: Function that creates the model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            model_name: Name of the model
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
        
        Returns:
            test_predictions, cv_score
        """
        print(f"\nTraining {model_name} with cross-validation...")
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        test_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
            print(f"\n  Fold {fold}/{self.n_folds}")
            
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Create model
            model = model_fn(input_dim=X_train.shape[1])
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=0
                )
            ]
            
            # Train
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_pred = model.predict(X_val, verbose=0).flatten()
            fold_rmse = rmse(y_val, val_pred)
            cv_scores.append(fold_rmse)
            
            # Test predictions
            test_pred = model.predict(X_test, verbose=0).flatten()
            test_preds.append(test_pred)
            
            print(f"    Fold {fold} RMSE: {fold_rmse:.6f}")
        
        # Average results
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        test_predictions = np.mean(test_preds, axis=0)
        
        print(f"  Mean CV RMSE: {mean_cv_score:.6f} (+/- {std_cv_score:.6f})")
        
        return test_predictions, mean_cv_score, std_cv_score
    
    def train_all_dl_models(self, X_train, y_train, X_test, epochs: int = 100):
        """Train all deep learning models"""
        results = {}
        
        # MLP
        mlp_pred, mlp_score, mlp_std = self.train_with_cv(
            DeepLearningModels.create_mlp,
            X_train, y_train, X_test,
            'MLP_Deep',
            epochs=epochs
        )
        results['MLP_Deep'] = {
            'predictions': mlp_pred,
            'cv_score': mlp_score,
            'std': mlp_std
        }
        
        # ResNet
        resnet_pred, resnet_score, resnet_std = self.train_with_cv(
            DeepLearningModels.create_resnet,
            X_train, y_train, X_test,
            'ResNet_Tabular',
            epochs=epochs
        )
        results['ResNet_Tabular'] = {
            'predictions': resnet_pred,
            'cv_score': resnet_score,
            'std': resnet_std
        }
        
        # Wide & Deep
        wd_pred, wd_score, wd_std = self.train_with_cv(
            DeepLearningModels.create_wide_and_deep,
            X_train, y_train, X_test,
            'WideAndDeep',
            epochs=epochs
        )
        results['WideAndDeep'] = {
            'predictions': wd_pred,
            'cv_score': wd_score,
            'std': wd_std
        }
        
        return results
