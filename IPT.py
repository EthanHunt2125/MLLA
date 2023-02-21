import random
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def norm_data(data):
    for i in range(0, data.shape[1]):
        data[:,i] = preprocessing.normalize([data[:,i]])
    return data
def calc_Wx(W, x):
    return W.dot(x)
def norm_row(column):
    m = np.mean(column)+1
    b = min(column)
    w= max(column)
    for i in range(0, len(column)):
        column[i] = round((column[i]+(w-b)) / m ,6)
    return column
def norm_data2(data):
    for i in range(0, data.shape[1]):
        data[:,i] = norm_row(data[:,i])
    return data
def prediction(P):
    maxi = 0
    for i in range(0, len(P)):
        if (P[i] > P[maxi]):
            maxi = i
    return maxi

def update(W,x,y):
    P=prediction(calc_Wx(W,x))
    if(P!=y):
        for i in range(0,len(W)):
            if(i!=y):
                W[i,:]=-(.1)*np.gradient(W[i,:])+W[i,:]
            else:
                W[i,:] = (.1)*np.gradient(W[i,:]) + W[i,:]
    return W

def IVP(data,tz):
    X = data.data
    y = data.target
    plt.plot(X, y, "o")
    scaler = StandardScaler()
    scaler.fit(X)
    scaler.transform(X)
    X = norm_data2(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tz)

    plt.show()
    W=np.random.uniform(.25,.75,(3,len(X[0])))
    print(W)
    for i in range(0,len(X_train)):
        W=update(W,X_train[i],y_train[i])
    c=0
    print(W)
    for i in range(0,len(X_test)):
        p=prediction(calc_Wx(W,X_test[i]))
        print('Expected:', y_test[i], ' Observed:', p)
        if(y_test[i]==p):

            c+=1
    return c/len(X_test)



W = np.random.random((2, 3))
x = np.random.random((3, 1))
iris = datasets.load_iris()
print(IVP(iris,.2))