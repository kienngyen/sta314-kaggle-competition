import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

N_FOLDS = 10
RANDOM_STATE = 42
TARGET_COL = 'y'
ID_COL = 'id'

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

def main():
    print("="*80)
    print("BASELINE: Model Comparison (The 'Brute Force' Search)")
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

    # Define models
    models = [
        ('Linear Reg', LinearRegression()),
        ('Ridge', Ridge(random_state=RANDOM_STATE)),
        ('Lasso', Lasso(random_state=RANDOM_STATE)),
        ('KNN', KNeighborsRegressor()),
        ('Random Forest', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
        ('Extra Trees', ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE))
    ]
    
    print("\n[Comparing Models]")
    results = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        rmse_score = -np.mean(scores)
        results.append((name, rmse_score))
        print(f"  {name:<15} CV RMSE: {rmse_score:.5f}")
        
    # Identify Winner
    results.sort(key=lambda x: x[1])
    winner_name, winner_score = results[0]
    print(f"\nWINNER: {winner_name} (RMSE: {winner_score:.5f})")
    
    # Generate Best Submission
    print(f"\nGenerating submission with {winner_name}...")
    final_model = [m for n, m in models if n == winner_name][0]
    final_model.fit(X, y)
    preds = final_model.predict(X_test)
            
    save_submission(preds, test_ids, 'submission_ExtraTrees.csv')

if __name__ == "__main__":
    main()
