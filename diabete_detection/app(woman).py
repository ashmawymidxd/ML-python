import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset from a CSV file named 'insulin.csv'
data = pd.read_csv('insulin.csv')

# Split the data into features (X) and target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Plot the distribution of diabetes classes
diabetes_counts = data['Outcome'].value_counts()
plt.figure(figsize=(6, 4))
diabetes_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Diabetes Classes')
plt.xlabel('Diabetes Status')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred_prob = model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_pred_prob > 0.5).astype(int)

# Calculate training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {training_accuracy:.2f}')

# Make predictions on the testing set
y_test_pred_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Calculate testing accuracy
testing_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Testing Accuracy: {testing_accuracy:.2f}')

# Example new data for prediction
new_data = pd.DataFrame({
    'Pregnancies': [6],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50],
})

# Convert categorical variables to numerical using one-hot encoding
# Ensure that the new_data DataFrame has the same columns as X_train
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

# Make predictions on the new data
new_data_pred_prob = model.predict_proba(new_data)[:, 1]
new_data_pred = (new_data_pred_prob > 0.5).astype(int)

# Print the predicted diabetes status for the new data
print(f'The predicted diabetes status is: {new_data_pred[0]}')
