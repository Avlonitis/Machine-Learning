import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from matplotlib.pyplot import plot,subplot
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

lowestMaeN = 0.0
lowestMaeN = 0.0

epochs = 50000
for N in [5,10,20,30,40,50]:
    meanmae = 0.0
    meanmse = 0.0
    for i in range(1,10):
        xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
        model = MLPRegressor(hidden_layer_sizes=N,
                              activation='relu',
                              solver='adam',
                              learning_rate='constant',
                              max_iter=epochs,
                              learning_rate_init=0.01)
        model.fit(xTrain,tTrain)
        predict =  model.predict(xTest)
        
        meanmse += regevaluate(tTest,predict,'mse')
        meanmae += regevaluate(tTest,predict,'mae')
        
    meanmse/=9
    meanmae/=9
    if(lowestMeanmae>meanmae):
        lowestMeanmae = meanmae
        lowestMaeN = N
    if(lowestMeanmse>meanmse):
        lowestMeanmse = meanmse   
        lowestMseN = N           

print(lowestMseN)        
print("Mean MSE: {}, N: {}".format(lowestMeanmse,lowestMseN))
print("Mean MAE: {}, N: {}".format(lowestMeanmae,lowestMaeN))
print("-------------------------")



xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
model = MLPRegressor(hidden_layer_sizes=lowestMseN,
                      activation='relu',
                      solver='adam',
                      learning_rate='constant',
                      max_iter=epochs,
                      learning_rate_init=0.01)
model.fit(xTrain,tTrain)    
predict =  model.predict(xTest)

subplot(1,1,1)
plot(predict,'r.',markersize=6)
plot(tTest,'b-',markersize=6)