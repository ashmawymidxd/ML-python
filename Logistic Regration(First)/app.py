import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix    

data = pd.read_csv("C:\\Users\\1\\Desktop\\ML\\Logistic Regration(First)\\diabetes.csv")
data.head()
data.shape
# print(data.shape)
data.Outcome.value_counts()
x = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].values
y = data["Outcome"].values

plt.figure(figsize=(10,10))
plt.scatter(x[y==0][:,0],x[y==0][:,1],color="blue",label="1")
plt.scatter(x[y==1][:,0],x[y==1][:,1],color="red",label="0")
plt.xlabel("Glucose")
plt.ylabel("Age")
plt.legend()
# plt.show()

class LogisticRegration:
    def __init__(self,l_rate=0.01,itrations=1000):
        self.l_rate = l_rate
        self.itrations = itrations

    # sigmoid function
    def fit(self,x,y):
        self.losses = []
        self.theta = np.zeros(1+x.shape[1])
        n = x.shape[0]

        for i in range(self.itrations):
            # linear model step 1
            y_pred = self.theta[0] + np.dot(x,self.theta[1:])
            z = y_pred
            # sigmoid function step 2
            g_z = 1/(1+np.e**(-z))   
            # cost function step 3
            cost = (-y*np.log(g_z)-(1-y)*np.log(1-g_z))/n
            self.losses.append(cost)
            # derivative step 4
            d_theta1 = (1/n)*np.dot(x.T,(g_z-y))
            d_theta0 = (1/n)*np.sum(g_z-y)
            # update step 5
            self.theta[1:] -= self.l_rate*d_theta1
            self.theta[0] -= self.l_rate*d_theta0
        return self
    
    # predict function
    def predict(self,x):
        y_pred = self.theta[0] + np.dot(x,self.theta[1:])
        z = y_pred
        g_z = 1/(1+np.e**(-z))
        return np.array([1 if i > 0.5 else 0 for i in g_z])

# features scaling using z-score
def scale(x):
    x_scaled = x - np.mean(x,axis=0)
    x_scaled /= np.std(x_scaled,axis=0)
    return x_scaled

# call the function scale()
x_sd = scale(x)
model = LogisticRegration()
model.fit(x_sd,y)

# print theta 0,1,2
print("theta 0: ",model.theta[0])
print("theta 1: ",model.theta[1])
print("theta 2: ",model.theta[2])

theta_0 = model.theta[0]
theta_1 = model.theta[1]
theta_2 = model.theta[2]

y_pred = model.predict(x_sd)

# compute confusion matrix
CM = confusion_matrix(y_pred,y,labels=[1,0])
print(CM)

TP = CM[0][0]
TN = CM[1][1]
FP = CM[0][1]
FN = CM[1][0]

print("---------------------------------")
ACC = (TP+TN)/(TP+TN+FP+FN)
print("ACC: ",ACC)
print("---------------------------------")
REC = TP/(TP+FN)
print("REC: ",REC)
print("---------------------------------")
F1 = 2*(REC*ACC)/(REC+ACC)
print("F1: ",F1)
print("---------------------------------")
PRE = TP/(TP+FP)
print("PRE: ",PRE)
print("---------------------------------")