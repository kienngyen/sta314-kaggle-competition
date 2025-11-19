"""
Data preprocessing and feature engineering module
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Advanced feature engineering for tabular data"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_polynomial_features(self, df: pd.DataFrame, 
                                   degree: int = 2, 
                                   top_k: int = 20) -> pd.DataFrame:
        """
        Create polynomial and interaction features for top correlated features
        
        Args:
            df: Input dataframe
            degree: Polynomial degree
            top_k: Number of top features to use for interactions
        
        Returns:
            DataFrame with new features
        """
        new_df = df.copy()
        feature_cols = [col for col in df.columns if col not in ['y', 'id']]
        
        # Create squared features for top_k features
        for col in feature_cols[:min(top_k, len(feature_cols))]:
            new_df[f'{col}_squared'] = df[col] ** 2
            new_df[f'{col}_cubed'] = df[col] ** 3
            new_df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            
        return new_df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregation features
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with statistical features
        """
        new_df = df.copy()
        feature_cols = [col for col in df.columns if col not in ['y', 'id']]
        feature_data = df[feature_cols]
        
        # Row-wise statistics
        new_df['row_mean'] = feature_data.mean(axis=1)
        new_df['row_std'] = feature_data.std(axis=1)
        new_df['row_min'] = feature_data.min(axis=1)
        new_df['row_max'] = feature_data.max(axis=1)
        new_df['row_median'] = feature_data.median(axis=1)
        new_df['row_range'] = new_df['row_max'] - new_df['row_min']
        new_df['row_skew'] = feature_data.skew(axis=1)
        new_df['row_kurt'] = feature_data.kurtosis(axis=1)
        
        # Count features
        new_df['row_num_zeros'] = (feature_data == 0).sum(axis=1)
        new_df['row_num_positive'] = (feature_data > 0).sum(axis=1)
        new_df['row_num_negative'] = (feature_data < 0).sum(axis=1)
        
        return new_df
    
    def create_cluster_features(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Create cluster-based features using KMeans
        
        Args:
            df: Input dataframe
            n_clusters: Number of clusters
        
        Returns:
            DataFrame with cluster features
        """
        from sklearn.cluster import KMeans
        
        new_df = df.copy()
        feature_cols = [col for col in df.columns if col not in ['y', 'id']]
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        new_df['cluster'] = kmeans.fit_predict(df[feature_cols])
        
        # Distance to cluster centers
        distances = kmeans.transform(df[feature_cols])
        for i in range(n_clusters):
            new_df[f'dist_to_cluster_{i}'] = distances[:, i]
        
        new_df['min_cluster_dist'] = distances.min(axis=1)
        
        return new_df
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data with feature engineering
        
        Args:
            X: Input features as numpy array
        
        Returns:
            Engineered features as numpy array
        """
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
        
        # Apply feature engineering
        df = self.create_polynomial_features(df, top_k=10)
        df = self.create_statistical_features(df)
        df = self.create_cluster_features(df, n_clusters=5)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        return df.values
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data with same feature engineering
        
        Args:
            X: Input features as numpy array
        
        Returns:
            Engineered features as numpy array
        """
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
        
        # Apply same transformations
        df = self.create_polynomial_features(df, top_k=10)
        df = self.create_statistical_features(df)
        df = self.create_cluster_features(df, n_clusters=5)
        
        return df.values


class DataPreprocessor:
    """Complete preprocessing pipeline"""
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 apply_pca: bool = False,
                 n_components: Optional[int] = None,
                 feature_selection: bool = False,
                 k_best: int = 50):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust', 'minmax', 'power')
            apply_pca: Whether to apply PCA
            n_components: Number of PCA components
            feature_selection: Whether to apply feature selection
            k_best: Number of best features to select
        """
        self.scaler_type = scaler_type
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.feature_selection = feature_selection
        self.k_best = k_best
        
        # Initialize transformers
        self.scaler = self._get_scaler()
        self.pca = None
        self.selector = None
        self.feature_engineer = FeatureEngineer()
        
    def _get_scaler(self):
        """Get scaler based on type"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Optional[pd.Series] = None,
                     engineer_features: bool = True) -> np.ndarray:
        """
        Fit and transform training data
        
        Args:
            X: Feature dataframe
            y: Target variable (optional)
            engineer_features: Whether to create engineered features
        
        Returns:
            Transformed feature array
        """
        X_transformed = X.copy()
        
        # Feature engineering
        if engineer_features:
            print("Creating engineered features...")
            X_transformed = self.feature_engineer.create_statistical_features(X_transformed)
            X_transformed = self.feature_engineer.create_polynomial_features(X_transformed, top_k=10)
            X_transformed = self.feature_engineer.create_cluster_features(X_transformed, n_clusters=5)
            print(f"Features after engineering: {X_transformed.shape[1]}")
        
        # Remove id column if present
        if 'id' in X_transformed.columns:
            X_transformed = X_transformed.drop('id', axis=1)
        
        # Handle missing values
        X_transformed = X_transformed.fillna(X_transformed.median())
        
        # Feature selection
        if self.feature_selection and y is not None:
            print(f"Selecting top {self.k_best} features...")
            self.selector = SelectKBest(score_func=f_regression, k=min(self.k_best, X_transformed.shape[1]))
            X_transformed = self.selector.fit_transform(X_transformed, y)
            print(f"Features after selection: {X_transformed.shape[1]}")
        else:
            X_transformed = X_transformed.values
        
        # Scaling
        print(f"Applying {self.scaler_type} scaling...")
        X_transformed = self.scaler.fit_transform(X_transformed)
        
        # PCA
        if self.apply_pca:
            n_comp = self.n_components or min(50, X_transformed.shape[1])
            print(f"Applying PCA with {n_comp} components...")
            self.pca = PCA(n_components=n_comp, random_state=42)
            X_transformed = self.pca.fit_transform(X_transformed)
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame, engineer_features: bool = True) -> np.ndarray:
        """
        Transform test data using fitted transformers
        
        Args:
            X: Feature dataframe
            engineer_features: Whether to create engineered features
        
        Returns:
            Transformed feature array
        """
        X_transformed = X.copy()
        
        # Feature engineering (same as training)
        if engineer_features:
            X_transformed = self.feature_engineer.create_statistical_features(X_transformed)
            X_transformed = self.feature_engineer.create_polynomial_features(X_transformed, top_k=10)
            X_transformed = self.feature_engineer.create_cluster_features(X_transformed, n_clusters=5)
        
        # Remove id column if present
        if 'id' in X_transformed.columns:
            X_transformed = X_transformed.drop('id', axis=1)
        
        # Handle missing values
        X_transformed = X_transformed.fillna(X_transformed.median())
        
        # Feature selection
        if self.selector is not None:
            X_transformed = self.selector.transform(X_transformed)
        else:
            X_transformed = X_transformed.values
        
        # Scaling
        X_transformed = self.scaler.transform(X_transformed)
        
        # PCA
        if self.pca is not None:
            X_transformed = self.pca.transform(X_transformed)
        
        return X_transformed


def prepare_data(train_df: pd.DataFrame, 
                test_df: pd.DataFrame,
                preprocessor: Optional[DataPreprocessor] = None,
                engineer_features: bool = True) -> Tuple:
    """
    Prepare data for modeling
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        preprocessor: DataPreprocessor instance
        engineer_features: Whether to engineer features
    
    Returns:
        X_train, y_train, X_test, test_ids
    """
    # Separate features and target
    y_train = train_df['y'].values
    X_train = train_df.drop('y', axis=1)
    X_test = test_df.copy()
    
    # Store test IDs
    test_ids = X_test['id'].values if 'id' in X_test.columns else np.arange(len(X_test))
    
    # Initialize preprocessor if not provided
    if preprocessor is None:
        preprocessor = DataPreprocessor(
            scaler_type='robust',
            feature_selection=False,
            apply_pca=False
        )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train, y_train, engineer_features=engineer_features)
    X_test_processed = preprocessor.transform(X_test, engineer_features=engineer_features)
    
    print(f"\nProcessed data shapes:")
    print(f"X_train: {X_train_processed.shape}")
    print(f"X_test: {X_test_processed.shape}")
    
    return X_train_processed, y_train, X_test_processed, test_ids, preprocessor
