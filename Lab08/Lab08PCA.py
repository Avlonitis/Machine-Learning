import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot,subplot
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

data = np.load('mnist_49.npz')

x = data['x']
t = data['t']


scoreTest = 0.0
scoreTrain = 0.0
for i in range(1,11):
    xTrain, xTest, tTrain, tTest = train_test_split(x,t,test_size=0.1)
    model = GaussianNB()
    model.fit(xTrain,tTrain)
    scoreTest += model.score(xTest,tTest)
    scoreTrain += model.score(xTrain,tTrain)
        

scoreTest /=10
scoreTrain /=10        
print("Mean Train accuracy: {}".format(scoreTest))
print("Mean Test accuracy: {}".format(scoreTrain))


acc_train = []
acc_test = []

mpla = [1,2,5,10,20,30,40,50,100,200]
for i in mpla:
    pca = PCA(n_components = i)
    x_pca = pca.fit_transform(x)
    meanTestScore = 0.0
    meanTrainScore = 0.0
    for i in range(1,11):
        xTrain, xTest, tTrain, tTest = train_test_split(x_pca,t,test_size=0.1)
        model = GaussianNB()
        model.fit(xTrain,tTrain)
        meanTestScore += model.score(xTest,tTest)
        meanTrainScore += model.score(xTrain,tTrain)
    meanTestScore/=10
    meanTrainScore/=10
    acc_test.append(meanTestScore)
    acc_train.append(meanTrainScore)

subplot(1,1,1)
plot(mpla,acc_train,'r-',markersize=6)
plot(mpla,acc_test,'b-',markersize=6)