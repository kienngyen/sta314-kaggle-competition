# STA314 Kaggle Competition: The Journey

This repository contains the code and strategies used to achieve top performance in the STA314 Kaggle Competition.

## Project Structure

*   **`code/`**: Contains the model scripts.
    *   `01_baseline.py`: Initial baseline models.
    *   `02_target_transform.py`: Exploration of target transformations.
    *   `03_vrm.py`: **The Final Winning Model** (Vicinal Risk Minimization).
*   **`data/`**: Contains the dataset (`trainingdata.csv`, `test_predictors.csv`).
*   **`submissions/`**: Contains the output CSV files.

## Quick Start: Reproducing the Winning Solution

### 1. Environment Setup

Ensure you have **Python 3.12** installed. We recommend using a virtual environment.

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
# Activate it (Windows)
.venv\Scripts\activate
# Activate it (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generating the Final Prediction

To run the Vicinal Risk Minimization (VRM) model, which is our final winning strategy:

```bash
python code/03_vrm.py
```

This script will:
1.  Load the data.
2.  Perform Dual Feature Selection (Lasso + CatBoost).
3.  Generate Interaction Features.
4.  Train an ensemble of CatBoost, XGBoost, and ExtraTrees using VRM (Noise Injection).
5.  Stack the models using Ridge Regression.
6.  Save the final predictions to `submissions/submission_research.csv`.

### 3. Submission

The file `submissions/submission_research.csv` is ready for submission to Kaggle. You can upload this file directly to the competition leaderboard.
