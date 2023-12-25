from flask import Flask, request, jsonify

app = Flask(__name__)

# Your linear regression parameters
theta_0 = 2.48  # Update with your calculated values
theta_1 = 9.78  # Update with your calculated values

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input hours from the request
        hours = float(request.form.get('hours'))

        # Use the linear regression equation to make a prediction
        prediction = theta_0 + theta_1 * hours

        # Return the predicted score as JSON
        return jsonify({'predicted_score': round(prediction, 2)})
    except Exception as e:
        # Handle the exception, you can log the error or return an error response
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
