import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import catboost as cb

# Load the preprocessed data and the tuned model
train_data = pd.read_csv('data/preprocessed_train.csv')
test_data = pd.read_csv('data/preprocessed_test.csv')
tuned_model = joblib.load('models/catboost_model_tuned.pkl')

# Split data into train and validation sets
X = train_data.drop(columns=['Exited'])
y = train_data['Exited']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the tuned model
y_val_prob_tuned = tuned_model.predict_proba(X_val)[:, 1]
auc_score_tuned = roc_auc_score(y_val, y_val_prob_tuned)
print(f"CatBoost Validation AUC after hyperparameter tuning: {auc_score_tuned:.4f}")

# Function to plot ROC curve
def plot_roc_curve(y_val, y_val_prob, model_name, linestyle='-', lw=2):
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, linestyle=linestyle, label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot ROC curve for the tuned model
plt.figure()
plot_roc_curve(y_val, y_val_prob_tuned, 'CatBoost Tuned', linestyle='--', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CatBoost Tuned')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix for the tuned model
y_val_pred_tuned = tuned_model.predict(X_val)
cm = confusion_matrix(y_val, y_val_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - CatBoost Tuned')
plt.show()

# Analysis of feature importance for the tuned model
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tuned_model.get_feature_importance()
})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances - CatBoost Tuned')
plt.show()

# Predict on test data
test_prob = tuned_model.predict_pro
