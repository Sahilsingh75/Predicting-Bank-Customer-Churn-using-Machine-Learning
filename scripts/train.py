import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib

# Load the preprocessed data
train_data = pd.read_csv('data/preprocessed_train.csv')
test_data = pd.read_csv('data/preprocessed_test.csv')

# Split data into train and validation sets
X = train_data.drop(columns=['Exited'])
y = train_data['Exited']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train basic models
xgb_model = xgb.XGBClassifier(random_state=42)
cb_model = cb.CatBoostClassifier(random_seed=42, verbose=0)
lgb_model = lgb.LGBMClassifier(random_state=42)

models = {
    'XGBoost': xgb_model,
    'CatBoost': cb_model,
    'LightGBM': lgb_model
}

auc_scores = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_prob)
    auc_scores[model_name] = auc_score
    print(f"{model_name} Validation AUC: {auc_score}")

# Identify the best model
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]

# Hyperparameter tuning for the best model (CatBoost)
param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [3, 5, 7]
}

grid = GridSearchCV(best_model, param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)
tuned_model = grid.best_estimator_

# Save the tuned model
tuned_model.save_model(f'models/{best_model_name.lower()}_model_tuned.cbm')
joblib.dump(tuned_model, f'models/{best_model_name.lower()}_model_tuned.pkl')
