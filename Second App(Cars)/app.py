import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\1\\Desktop\\ML\\Second App(Cars)\\cars.csv")
data.head()

data.shape

x = data[['enginesize','carheight']].values
y=data['price'].values

print (x.shape)
print (y.shape)

class LinearRegression:
    #constructor
    def __init__(self,l_rate=0.001,iterations=1000):
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
    
#fetuer scaling using z-score
def scale(x):
    x_scaled = x-np.mean(x,axis=0)
    x_scaled = x_scaled/np.std(x_scaled,axis=0)
    return x_scaled

#call the function scale()
lr = LinearRegression()
lr.fitGD(scale(x),y)
lr.fitNQ(scale(x),y)

#print the theta values 0,1,2 for gradient descent
print('theta_0 = ',lr.theta[0])
print('theta_1 = ',lr.theta[1])
print('theta_2 = ',lr.theta[2])

#print the theta values 0,1,2 for normal equation
print('theta_0 = ',lr.thetas[0])
print('theta_1 = ',lr.thetas[1])
print('theta_2 = ',lr.thetas[2])

#predect traning examples using gradient descent
y_pred = lr.predictGD(scale(x))

#predect traning examples using normal equation
y_predNQ = lr.predictNQ(scale(x))

#evaluation traning modelGD usinng adjusted R^2
errors = np.sum((y_pred-y)**2)
sst = np.sum((y-np.mean(y))**2)
r2_GD = 1-(errors/sst)

adja_r2_GD = 1-((1-r2_GD)*(x.shape[0]-1)/(x.shape[0]-x.shape[1]-1))
print(adja_r2_GD)

#evaluation traning normal equation usinng adjusted R^2
errors = np.sum((y_predNQ-y)**2)
sst = np.sum((y-np.mean(y))**2)
r2_NQ = 1-(errors/sst)

adja_r2_NQ = 1-((1-r2_NQ)*(x.shape[0]-1)/(x.shape[0]-x.shape[1]-1))
print(adja_r2_NQ)

#try the model on new data with gradient descent
x_new = np.array([[130,50],[150,60],[200,70],[250,80]])
y_new = lr.predictGD(scale(x_new))
print(y_new)

#try the model on new data with normal equation
y_new = lr.predictNQ(scale(x_new))
print(y_new)
