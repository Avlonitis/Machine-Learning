import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot,subplot


data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None).values

NoP = data.shape[0]
NoA = data.shape[1]

map_dict = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':0}

x = data[:,0:4]
t = np.zeros(NoP)

for pattern in range(NoP):
    t[pattern] = map_dict[data[pattern,4]]

for i in range(1,10):
    xTrain, xTest, tTrain, yTest = train_test_split(x,t,test_size=0.1)
    subplot(3,3,i)
    plot(xTrain[:,0],xTrain[:,2],'bo')
    plot(xTest[:,0],xTest[:,2],'ro')
