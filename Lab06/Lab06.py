import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot,subplot
from scipy.stats import norm

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

def nbtrain(x,t):
    
    x0 = x[t == 0,:]
    x1 = x[t == 1,:]
    
    priorX0 = len(x0)/len(t)
    priorX1 = len(x1)/len(t)
    
    m = np.zeros((2,len(x)))
    s = np.zeros((2,len(x)))
    for i in range(4):
        m[0,i] = np.mean(x0[:,i])
        s[0,i] = np.std(x0[:,i])
        m[1,i] = np.mean(x1[:,i])
        s[1,i] = np.std(x1[:,i]) 
    
    model = {'prior':[priorX0,priorX1],'mu':m,'sigma':s}
    
    return model
def nbpredict(x, model):
    predict = np.zeros(len(x))
    for p in range(len(x)):
        L = model.get("prior")[1]/model.get("prior")[0]
        for i in range(len(x[p])):
            L *= norm.pdf(x[p,i],model.get("mu")[1,i],model.get("sigma")[1,i])/norm.pdf(x[p,i],model.get("mu")[0,i],model.get("sigma")[0,i])
        if(L<1):
            predict[p] = 0
        elif(L>1):
            predict[p] = 1
            
    return predict
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None).values

NoP = data.shape[0]
NoA = data.shape[1]

map_dict = {'Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':0}

x = data[:,0:4].astype(float)
t = np.zeros(NoP)
#to kanw numpy array giati diaforetika den leitourgei to x[t==0,:]
t = np.array(list(map(lambda d: map_dict[d[4]],data)))

meanaccuracy = 0.0
meanprecision = 0.0
meanrecall = 0.0
meanfmeasure = 0.0
meansensitivity = 0.0
meanspecificity = 0.0
for i in range(1,10):
    xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
    
    model = nbtrain(xTrain,tTrain)
    predict = nbpredict(xTest,model)
        
    meanaccuracy += evaluate(tTest, predict,'accuracy')
    meanprecision += evaluate(tTest, predict,'precision')
    meanrecall += evaluate(tTest, predict,'recall')
    meanfmeasure += evaluate(tTest, predict,'fmeasure')
    meansensitivity += evaluate(tTest, predict,'sensitivity')
    meanspecificity += evaluate(tTest, predict,'specificity')
    
    subplot(3,3,i)
    plot(predict,'ro',markersize=6)
    plot(tTest,'b.',markersize=6)
   

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
