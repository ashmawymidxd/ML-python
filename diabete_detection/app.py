import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Assuming your dataset is in a CSV file named 'diabetes_data.csv'
data = pd.read_csv('diabetes.csv')

# Split the data into features and target variable
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Count the occurrences of each class in the 'diabetes' column
diabetes_counts = data['diabetes'].value_counts()

# Plot a bar chart
plt.figure(figsize=(6, 4))
diabetes_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Diabetes Classes')
plt.xlabel('Diabetes Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
# plt.show()

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred_prob = model.predict_proba(X_train)[:, 1]  # Use probabilities for accuracy calculation

# Convert probabilities to binary predictions (0 or 1)
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

#Female,44.0,0,0,never,19.31,6.5,200,1
#Male,67.0,0,1,not current,27.32,6.5,200,1 

# Assuming you have new data in the same format as your training data
new_data = pd.DataFrame({
    'gender': ['Male'],
    'age': [67.0],
    'hypertension': [0],
    'heart_disease': [1],
    'smoking_history': ['not current'],
    'bmi': [27.32],
    'HbA1c_level': [6.5],
    'blood_glucose_level': [200]
})

# Convert categorical variables to numerical using one-hot encoding
# Ensure that the new_data DataFrame has the same columns as X_train
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

# Make predictions on the new data
new_data_pred_prob = model.predict_proba(new_data)[:, 1]
new_data_pred = (new_data_pred_prob > 0.5).astype(int)

# Print the results
print(f'The predicted diabetes status is: {new_data_pred[0]}')

# referance mode code to detect diabetes: 
# https://www.kaggle.com/code/shrutimechlearn/step-by-step-diabetes-classification-knn-detailed/input

# referance mode code to Prediction  diabetes:
# https://www.kaggle.com/code/yogidsba/diabetes-prediction-eda-model/notebook