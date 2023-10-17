import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the training data from a CSV file
train_data = pd.read_csv('Train.csv')

# Split the data into features and target
X = train_data.drop(['INCIDENT_ID', 'DATE', 'MALICIOUS_OFFENSE'], axis=1)
y = train_data['MALICIOUS_OFFENSE']

# Fill missing values with the mean of the corresponding column
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Print the training score of the model
print("Training score:", accuracy_score(y, clf.predict(X)))

# Load the test data from a CSV file
test_data = pd.read_csv('Test.csv')

# Fill missing values with the mean of the corresponding column
X_test = test_data.drop(['INCIDENT_ID', 'DATE'], axis=1)
X_test = imputer.transform(X_test)

# Predict the target for the test data
y_pred = clf.predict(X_test)

# Save the predicted target values in the format for the sample submission file
submission = pd.DataFrame({'INCIDENT_ID': test_data['INCIDENT_ID'], 'MALICIOUS_OFFENSE': y_pred})
submission.to_csv('submission.csv', index=False)
