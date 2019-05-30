import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot,subplot
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error

def regevaluate(t, predict, criterion):
    if(criterion=='mse'):
        return mean_squared_error(t,predict)
    else:
        return mean_absolute_error(t,predict)
    
data = pd.read_csv("housing.data",header=None,sep="\s+").values

NoP = data.shape[0]
NoA = data.shape[1] 


x = data[:,0:NoA-1]
t = np.zeros(NoP)

t = (list(map(lambda d: d[NoA-1],data)))


lowestMeanmse = 1000.0
lowestMeanmae = 1000.0
lowestMaeGamma = 0.0
lowestMaeC = 0.0
lowestMseGamma = 0.0
lowestMseC = 0.0

for g in [0.0001, 0.001, 0.01, 0.1]:
    for c in [1, 10, 100, 1000]:
        meanmae = 0.0
        meanmse = 0.0
        for i in range(1,10):
            xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
            model = SVR(C=c, kernel='rbf',gamma=g)
            model.fit(xTrain,tTrain)
            predict = model.predict(xTest)
            
            meanmse += regevaluate(tTest,predict,'mse')
            meanmae += regevaluate(tTest,predict,'mae')
            
        meanmse/=9
        meanmae/=9
        if(lowestMeanmae>meanmae):
            lowestMeanmae = meanmae
            lowestMaeGamma = g
            lowestMaeC = c
        if(lowestMeanmse>meanmse):
            lowestMeanmse = meanmse
            lowestMseGamma = g
            lowestMseC = c                

        
print("Mean MSE: {}, gamma: {}, C: {}".format(lowestMeanmse,lowestMseGamma,lowestMseC))
print("Mean MAE: {}, gamma: {}, C: {}".format(lowestMeanmae,lowestMaeGamma,lowestMaeC))
print("-------------------------")



xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
model = SVR(C=lowestMseC, kernel='rbf',gamma=lowestMseGamma)
model.fit(xTrain,tTrain)
predict = model.predict(xTest)

subplot(1,1,1)
plot(predict,'r.',markersize=6)
plot(tTest,markersize=6)