from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset and pre-process
data = pd.read_csv('diabetes.csv')
X = pd.get_dummies(data.drop('diabetes', axis=1))
y = data['diabetes']
CORS(app)  # Enable CORS for all routes
# Endpoint to predict diabetes
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Load the model inside the route function
        model = LogisticRegression()
        model.fit(X, y)

        # Get input values from the request
        input_data = request.get_json()

        # Prepare the input data
        input_df = pd.DataFrame(input_data, index=[0])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Make predictions
        prediction = model.predict(input_df)

        # Prepare the response
        response = {'prediction': int(prediction[0])}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# 1)-Headers
# Content-Type = application/json

# 2)-Body
# -row
# {
#     "gender": "Female",
#     "age": 60.0,
#     "hypertension": 1,
#     "heart_disease": 0,
#     "smoking_history": "never",
#     "bmi": 28.5,
#     "HbA1c_level": 7.0,
#     "blood_glucose_level": 120
# }
