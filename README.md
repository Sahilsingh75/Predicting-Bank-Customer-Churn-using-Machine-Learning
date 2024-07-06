# Kaggle_hackathon - Predicting Bank Customer Churn
Predicting bank customer churn using machine learning models as part of a Kaggle hackathon. Includes data preprocessing, model training, evaluation, and hyperparameter tuning.

## Overview
This project was part of a Kaggle hackathon organized by our university. The goal was to predict whether a bank customer will leave (churn) based on various customer details.

## Project Structure
kaggle_hackathon/
    -data/
      -train.csv
      -test.csv
    -models/
      -final_model_tuned.cbm
    -scripts/
      -preprocess.py
      -train.py
      -evaluate.py
    -kaggle_hackathon.ipynb
    -submission.csv
    -README.md
    -requirements.txt


### Files

- `data/`: Contains the training and test data.
- `models/`: Contains the saved CatBoost model after tuning.
- `scripts/`: Contains Python scripts for data processing, model training, and evaluation.
- `kaggle_hackathon.ipynb`: The main Jupyter notebook for the project.
- `submission.csv`: The submission file for Kaggle.
- `README.md`: This file.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, `seaborn`, `joblib`

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/Sahilsingh75/Kaggle_hackathon.git
   cd kaggle_hackathon

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Running the Project
  Process the data:
  
  ```bash
  Copy code
  python scripts/preprocess.py

  Train the models:
  ```bash
  python scripts/train.py


  Evaluate the models and create the submission file:  
  ```bash
  python scripts/evaluate.py

Results
The CatBoost model performed the best with the highest AUC score. After tuning the hyperparameters, the performance improved even more.



