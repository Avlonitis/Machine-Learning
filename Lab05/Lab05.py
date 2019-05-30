import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib.pyplot import plot,subplot
from sklearn.svm import SVC

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
 
t = (list(map(lambda d: map_dict[d[4]],data)))

bestMeanTestAccuracy=-10
bestC=0
bestG=0

for gm in [0.01, 0.03, 0.1, 0.3, 1]:
    for cc in [1, 10, 100, 1000]:
        meanaccuracy = 0.0
        meanprecision = 0.0
        meanrecall = 0.0
        meanfmeasure = 0.0
        meansensitivity = 0.0
        meanspecificity = 0.0
        for i in range(1,10):
            xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
        
            clf = SVC(gamma=gm,C=cc,kernel='rbf')
            clf.fit(xTrain,tTrain)
            
            y = clf.predict(xTest)
            meanaccuracy += evaluate(tTest, y,'accuracy')
            meanprecision += evaluate(tTest, y,'precision')
            meanrecall += evaluate(tTest, y,'recall')
            meanfmeasure += evaluate(tTest, y,'fmeasure')
            meansensitivity += evaluate(tTest, y,'sensitivity')
            meanspecificity += evaluate(tTest, y,'specificity')
            
#            subplot(3,3,i)
#            plot(tTest,'bo',markersize=6)
#            plot(y,'ro',markersize=5)

        meanaccuracy /= 9
        meanprecision /= 9
        meanrecall /= 9
        meanfmeasure /= 9
        meansensitivity /= 9
        meanspecificity /= 9
        if(meanaccuracy>bestMeanTestAccuracy):
            bestMeanTestAccuracy=meanaccuracy
            bestC = cc
            bestG = gm
        print("--------C:{}, gamma={}--------".format(gm,cc))
        print("Mean accuracy: "+str(meanaccuracy))
        print("Mean precision: "+str(meanprecision))
        print("Mean recall: "+str(meanrecall))
        print("Mean fmeasure: "+str(meanfmeasure))
        print("Mean sensitivity: "+str(meansensitivity))
        print("Mean specificity: "+str(meanspecificity))
        print(" ")
        
print("Best accuracy: {}, C: {}, gamma: {}".format(bestMeanTestAccuracy,bestC,bestG))