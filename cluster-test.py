import copy
from collections import Counter

from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from LstmModel import LstmModel

classNum = 8
clustervoteResultDict = {}
clustervoteIndexDict = {}
clusterTruthDict = {}
clustervoteLabelDict = {}
clusterSampleDataDict = {}
clusterIndexDict = {}

lstmModel = LstmModel()
lstmModel.loadModel()
# lstmModel.model.eval()
preTrainData, preTrainLabel, trainData, trainLabel, testData, testLabel = lstmModel.inputData()
lstmIndex, lstmResult = lstmModel.predict(testData)

indexs, preTrainData = lstmModel.featureExtract(preTrainData)
indexs, testData = lstmModel.featureExtract(testData)

clustering = DBSCAN(eps=0.5, min_samples=5)
# clustering.fit(preTrainData[:, 1:])
# result = clustering.fit_predict(testData[:, 1:])
clustering.fit(preTrainData)
result = clustering.fit_predict(testData)
print("dbscan accuracy: ", accuracy_score(testLabel, result))
print("lstm accuracy: ", accuracy_score(testLabel, lstmResult))
print(classification_report(testLabel, lstmResult, target_names=["AUDIO", "BROWSING", "CHAT", "FILE-TRANSFER", "MAIL", "VIDEO"], digits=4))

for item in range(classNum):
    clustervoteResultDict[item] = []
    clustervoteIndexDict[item] = []

for key in clusterTruthDict.keys():
    result = Counter(clusterTruthDict[key])
    clustervoteLabelDict[key] = max(result.keys(), key=result.get)
    for j in clusterSampleDataDict[key]:
        clustervoteResultDict[max(result.keys(), key=result.get)].append(copy.copy(j))
    for j in clusterIndexDict[key]:
        clustervoteIndexDict[max(result.keys(), key=result.get)].append(int(copy.copy(j)))
