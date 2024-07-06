```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Drop unnecessary columns
train_data = train_data.drop(columns=['CustomerId', 'Surname'])
test_data_with_id = test_data.copy()
test_data = test_data.drop(columns=['CustomerId', 'Surname'])

# Encode categorical variables
le = LabelEncoder()
train_data['Gender'] = le.fit_transform(train_data['Gender'])
test_data['Gender'] = le.transform(test_data['Gender'])

# One-hot encode Geography
train_data = pd.get_dummies(train_data, columns=['Geography'])
test_data = pd.get_dummies(test_data, columns=['Geography'])

# Ensure test data has the same columns as train data
missing_cols = set(train_data.columns) - set(test_data.columns)
for c in missing_cols:
    test_data[c] = 0
test_data = test_data[train_data.columns.drop('Exited')]

# Scale features
scaler = StandardScaler()
features_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

# Save preprocessed data
train_data.to_csv('data/preprocessed_train.csv', index=False)
test_data.to_csv('data/preprocessed_test.csv', index=False)
