#!/bin/bash
# Quick start script for macOS

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "  KAGGLE COMPETITION - QUICK START (macOS)"
echo "================================================================================"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo -e "\n${BLUE}[1/3]${NC} Activating virtual environment..."
source venv/bin/activate

# Export PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run training
echo -e "\n${BLUE}[2/3]${NC} Training models (XGBoost + LightGBM with 3-fold CV)..."
python quick_train.py --models xgboost lightgbm --n_folds 3

# Check results
echo -e "\n${BLUE}[3/3]${NC} Results:"
if [ -f "submissions/submission_quick_ensemble.csv" ]; then
    echo -e "${GREEN}✓${NC} Submission created: submissions/submission_quick_ensemble.csv"
    echo ""
    echo "Next steps:"
    echo "  1. Check model performance in the output above"
    echo "  2. Upload submissions/submission_quick_ensemble.csv to Kaggle"
    echo "  3. For full pipeline: python train_and_predict.py"
else
    echo "⚠️  Training may have encountered issues. Check output above."
fi

echo ""
echo "================================================================================"
