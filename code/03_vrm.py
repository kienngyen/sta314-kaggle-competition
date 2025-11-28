import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV, BayesianRidge, Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

N_FOLDS = 10
RANDOM_STATE = 42
TARGET_COL = 'y'
ID_COL = 'id'
NOISE_LEVEL = 0.01

def load_data():
    print("Loading data...")
    train_df = pd.read_csv('data/trainingdata.csv')
    test_df = pd.read_csv('data/test_predictors.csv')
    sample_submission = pd.read_csv('data/SampleSubmission.csv')
    return train_df, test_df, sample_submission

def save_submission(preds, test_ids, filename):
    submission = pd.DataFrame({'id': test_ids, 'y': preds})
    os.makedirs('submissions', exist_ok=True)
    path = os.path.join('submissions', filename)
    submission.to_csv(path, index=False)
    print(f"Submission saved to {path}")

class DualFeatureSelector:
    def __init__(self, n_lasso_features=20, n_tree_features=20):
        self.n_lasso = n_lasso_features
        self.n_tree = n_tree_features
        self.selected_indices = None
        
    def fit(self, X, y):
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000).fit(X_scaled, y)
        lasso_indices = np.argsort(np.abs(lasso.coef_))[-self.n_lasso:]
        
        cb = CatBoostRegressor(iterations=500, verbose=0, random_seed=RANDOM_STATE, allow_writing_files=False)
        cb.fit(X, y)
        tree_indices = np.argsort(cb.get_feature_importance())[-self.n_tree:]
        
        self.selected_indices = np.unique(np.concatenate([lasso_indices, tree_indices]))
        return self
        
    def transform(self, X):
        return X[:, self.selected_indices]

class InteractionMiner:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.top_indices = None
        
    def fit(self, X, y):
        rf = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE).fit(X, y)
        self.top_indices = np.argsort(rf.feature_importances_)[-self.top_k:]
        return self
        
    def transform(self, X):
        X_top = X[:, self.top_indices]
        X_interactions = self.poly.fit_transform(X_top)
        X_interactions = X_interactions[:, self.top_k:] 
        return np.hstack([X, X_interactions])

def augment_data(X, y, noise_level=0.01, n_copies=1):
    X_aug = [X]
    y_aug = [y]
    scaler = RobustScaler()
    scaler.fit(X)
    scale = scaler.scale_
    scale[scale == 0] = 1.0
    
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_level, X.shape) * scale
        X_new = X + noise
        X_aug.append(X_new)
        y_aug.append(y)
        
    return np.vstack(X_aug), np.concatenate(y_aug)

def main():
    print("="*80)
    print("Data Augmentation")
    print("="*80)
    train_df, test_df, sample_submission = load_data()
    y = train_df[TARGET_COL].values
    X = train_df.drop(TARGET_COL, axis=1).values
    if ID_COL in test_df.columns:
        test_ids = test_df[ID_COL].values
        X_test = test_df.drop(ID_COL, axis=1).values
    else:
        test_ids = sample_submission[ID_COL].values
        X_test = test_df.values

    # 1. Select Features (Final)
    selector = DualFeatureSelector(n_lasso_features=25, n_tree_features=25)
    selector.fit(X, y)
    X_sel = selector.transform(X)
    X_test_sel = selector.transform(X_test)
    print(f"\n[Step 1] Feature Selection: {X.shape[1]} -> {X_sel.shape[1]} features")
    print("  (Using Dual Selector: Lasso + CatBoost)")
    
    # 2. Add Interactions
    miner = InteractionMiner(top_k=8)
    miner.fit(X_sel, y)
    X_eng = miner.transform(X_sel)
    X_test_eng = miner.transform(X_test_sel)
    print(f"[Step 2] Feature Engineering: {X_sel.shape[1]} -> {X_eng.shape[1]} features")
    print("  (Added interactions for Top 8 features)")
    
    # 3. Models
    target_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE, n_quantiles=min(len(y), 1000))
    models = [
        ('res_cat', CatBoostRegressor(iterations=1500, learning_rate=0.01, depth=6, verbose=0, random_seed=RANDOM_STATE, allow_writing_files=False)),
        ('res_xgb', XGBRegressor(n_estimators=1500, learning_rate=0.01, max_depth=5, n_jobs=-1, random_state=RANDOM_STATE)),
        ('res_et', ExtraTreesRegressor(n_estimators=1000, max_depth=None, min_samples_split=5, n_jobs=-1, random_state=RANDOM_STATE)),
    ]
    wrapped_models = [(name, TransformedTargetRegressor(regressor=m, transformer=target_transformer)) for name, m in models]
    
    print("[Training with Noise Injection]")
    oof_preds = np.zeros((X_eng.shape[0], len(models)))
    test_preds = np.zeros((X_test_eng.shape[0], len(models)))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for i, (name, model) in enumerate(wrapped_models):
        print(f"  Training {name}...")
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_eng, y)):
            # Augment ONLY the training fold
            X_fold_train = X_eng[train_idx]
            y_fold_train = y[train_idx]
            X_fold_aug, y_fold_aug = augment_data(X_fold_train, y_fold_train, noise_level=NOISE_LEVEL, n_copies=1)
            
            model.fit(X_fold_aug, y_fold_aug)
            val_pred = model.predict(X_eng[val_idx])
            oof_preds[val_idx, i] = val_pred
            fold_scores.append(np.sqrt(mean_squared_error(y[val_idx], val_pred)))
            test_preds[:, i] += model.predict(X_test_eng) / N_FOLDS
        print(f"    {name} CV RMSE: {np.mean(fold_scores):.5f}")
        
    print("[Stacking]")
    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    meta_model.fit(oof_preds, y)
    final_pred = meta_model.predict(test_preds)
    
    save_submission(final_pred, test_ids, 'submission_research.csv')

if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
