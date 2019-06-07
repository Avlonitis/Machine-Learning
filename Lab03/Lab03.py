import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot,subplot

def perceptron(x,t,MAXEPOCHS,beta):
    w = np.random.randn(len(x[0]))
    u = 0.0
    
    for i in range(MAXEPOCHS):
        flag = 0
        for p in range(len(x)):
            u = x[p,:].dot(w)
            
            y = (0.0 if (u<0) else 1.0)
            
            if(t[p]!=y):
                w += beta*(t[p]-y)*x[p,:]
                flag = 1
        if(flag==0):
          break
          
    return w

def evaluate(t,predict,criterion):
    tn=float(0)
    fn=float(0)
    tp=float(0)
    fp=float(0)
    
    for i in range(len(t)):
        if(predict[i]==0):
            if(t[i]==0):
                tn+=1
            else:
                fn+=1
        else:
            if(t[i]==0):
                fp+=1
            else:
                tp+=1
                

    if(criterion=='accuracy'):
        if (tp+tn+fp+fn == 0):
            return 0
        return ((tp+tn)/(tp+tn+fp+fn))
    elif (criterion == 'precision'):
         if (tp+fp == 0):
             return 0
         return (tp/(tp+fp));
    elif (criterion == 'recall'):
         if (tp+fn == 0):
            return 0
         return (tp/(tp+fn))
    elif (criterion == 'fmeasure'):
        precision = evaluate(t,predict,'precision')
        recall = evaluate(t,predict,'recall')
        if (precision+recall == 0):
            return 0
        return ((precision*recall)/((precision+recall)/2))
    elif (criterion == 'sensitivity'):
        if (tp+fn == 0):
            return 0
        return (tp/(tp+fn))
    elif (criterion == 'specificity'):
        if (tn+fp == 0):
            return 0
        return (tn/(tn+fp))
    
      

 
 
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None).values

NoP = data.shape[0]
NoA = data.shape[1]

map_dict = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':0}
t_dict = {1.0: 1.0,0.0:-1.0}

x = data[:,0:4].astype(float)
t = np.zeros(NoP)
 
t = list(map(lambda d: map_dict[d[4]],data))
x =  np.hstack((x,np.ones([NoP,1])))
x = x.astype(float)

meanaccuracy = 0.0
meanprecision = 0.0
meanrecall = 0.0
meanfmeasure = 0.0
meansensitivity = 0.0
meanspecificity = 0.0

maxP = int(input('Dwse max plithos epoxwn: '))
b = float(input('Vima ekpaidefshs: '))
for i in range(1,10):
    xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
        
    w = perceptron(xTrain,tTrain,maxP,b) 
    y = xTest.dot(w)
    
    predict = np.array(list(map(lambda p: 0 if p<0 else 1,y)))
    
    meanaccuracy += evaluate(tTest,predict,'accuracy')
    meanprecision += evaluate(tTest,predict,'precision')
    meanrecall += evaluate(tTest,predict,'recall')
    meanfmeasure += evaluate(tTest,predict,'fmeasure')
    meansensitivity += evaluate(tTest,predict,'sensitivity')
    meanspecificity += evaluate(tTest,predict,'specificity')
    
    subplot(3,3,i)
    
    plot(predict,'ro',markersize=10)
    plot(tTest,'bo',markersize=5)



meanaccuracy /= 9
meanprecision /= 9
meanrecall /= 9
meanfmeasure /= 9
meansensitivity /= 9
meanspecificity /= 9

print("Mean accuracy: "+str(meanaccuracy))
print("Mean precision: "+str(meanprecision))
print("Mean recall: "+str(meanrecall))
print("Mean fmeasure: "+str(meanfmeasure))
print("Mean sensitivity: "+str(meansensitivity))
print("Mean specificity: "+str(meanspecificity))