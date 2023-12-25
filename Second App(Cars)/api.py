from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

class LinearRegression:
    #constructor
    def __init__(self,l_rate=0.01,iterations=1000):
        self.l_rate = l_rate
        self.itrations = iterations

    #fit function to train model py gradient descent
    def fitGD(self,x,y):
        self.cost = []
        self.theta = np.zeros((1+x.shape[1]))
        n = x.shape[0]
        for i in range(self.itrations):
            y_pred = self.theta[0] + np.dot(x,self.theta[1:])
            mse = 1/n * np.sum((y_pred-y)**2)
            self.cost.append(mse)
            #derivative
            d_theta1 = 2/n * np.dot(x.T,(y_pred-y))
            d_theta0 = 2/n * np.sum(y_pred-y)
            #value update
            self.theta[1:] -= self.l_rate * d_theta1
            self.theta[0] -= self.l_rate * d_theta0
        return self

    #predict function
    def predictGD(self,x):
        return self.theta[0] + np.dot(x,self.theta[1:])

    #fit function to train model py normal equation
    def fitNQ(self,x,y):
        z = np.ones((x.shape[0],1))
        x = np.append(z,x,axis=1)
        self.thetas = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y) #normal equation
        return self

    #predict function for normal equation
    def predictNQ(self,x):
        z = np.ones((x.shape[0],1))
        x = np.append(z,x,axis=1)
        return np.dot(x,self.thetas)


# Load the data
file_path = "C:\\Users\\1\\Desktop\\ML\\Second App(Cars)\\cars.csv"
data = pd.read_csv(file_path)

# Feature scaling using z-score
def scale(x):
    x_scaled = x - np.mean(x, axis=0)
    x_scaled = x_scaled / np.std(x_scaled, axis=0)
    return x_scaled

# Train the model
x = data[['enginesize', 'carheight']].values
y = data['price'].values
lr = LinearRegression()
lr.fitGD(scale(x), y)
lr.fitNQ(scale(x), y)

# Flask route to predict using gradient descent
@app.route('/predict_gd', methods=['POST'])
def predict_gd():
    content = request.get_json()
    x_new = np.array(content['features'])
    x_scaled = scale(x_new)
    y_pred = lr.predictGD(x_scaled)
    return jsonify({'prediction': y_pred.tolist()})

# Flask route to predict using normal equation
@app.route('/predict_nq', methods=['POST'])
def predict_nq():
    content = request.get_json()
    x_new = np.array(content['features'])
    x_scaled = scale(x_new)
    y_pred = lr.predictNQ(x_scaled)
    return jsonify({'prediction': y_pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
