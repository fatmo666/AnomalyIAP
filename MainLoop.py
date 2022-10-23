import math
from copy import copy, deepcopy
from random import shuffle

import more_itertools
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
    classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from DC import DC
from LstmModel import LstmModel, LstmModel02, LstmModel03
import numpy as np

from lime.lime_tabular import LimeTabularExplainer
from interpret.ext.blackbox import PFIExplainer

from utils import Normalizer
import pandas as pd

batchSize = 32
distanceThreshold = 10
simThreshold = 0.2
topFeatureNum = 8

targets = ['AUDIO', 'BROWSING', 'CHAT', 'FILE-TRANSFER', 'MAIL', 'VIDEO', 'VOIP', 'P2P']
features_names = ['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Protocol','Timestamp','Flow Duration','Total Fwd Packet','Total Bwd packets','Total Length of Fwd Packet','Total Length of Bwd Packet','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Packet Length Min','Packet Length Max','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Fwd Segment Size Avg','Bwd Segment Size Avg','Fwd Bytes/Bulk Avg','Fwd Packet/Bulk Avg','Fwd Bulk Rate Avg','Bwd Bytes/Bulk Avg','Bwd Packet/Bulk Avg','Bwd Bulk Rate Avg','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','FWD Init Win Bytes','Bwd Init Win Bytes','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']
# features_names = [0 for i in range(79)]

lstmModel = LstmModel()
lstmModel02 = LstmModel02()
lstmModel03 = LstmModel03()
dc = DC(eps=0.5, min_samples=4, lstmModel=lstmModel)
dc02 = DC(eps=0.2, min_samples=4, lstmModel=lstmModel)
dc03 = DC(eps=0.8, min_samples=4, lstmModel=lstmModel)

def getEvaluation(resultTruth, resultTest):
    print("accuracy: ", accuracy_score(resultTruth, resultTest))
    print("f1_score-macro: ", f1_score(resultTruth, resultTest, average='macro'))
    print("f1_score-micro: ", f1_score(resultTruth, resultTest, average='micro'))
    print("recall: ", recall_score(resultTruth, resultTest, average='macro'))
    print("precision: ", precision_score(resultTruth, resultTest, average='macro'))

    cnf_matrix = confusion_matrix(resultTruth, resultTest)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    print("TP: ", TP)
    print("FP: ", FP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print("TPR:", np.mean(TPR))
    print("FPR:", np.mean(FPR))
    print("FNR:", np.mean(FNR))

    try:
        print(classification_report(resultTruth, resultTest,
                                    target_names=["AUDIO", "BROWSING", "CHAT", "FILE-TRANSFER", "MAIL", "VIDEO", "VOIP",
                                                  "P2P"], digits=4))
    except:
        pass
    print("all TPR:", TPR)
    print("all FPR:", FPR)

def simCheck(localex, globalex):
    temp = 0
    for i in localex:
        if i in globalex:
            temp = temp + 1
    similarity = temp / (len(localex) + len(globalex))
    similarity = similarity * 2
    return similarity

def localFeatureExtract(expList):
    featureList = []
    for item in expList:
        splitResult = item[0].split('<')
        if len(splitResult) == 2:
            featureList.append(splitResult[0])
        elif len(splitResult) == 3:
            featureList.append(splitResult[1])
        elif len(splitResult) == 1:
            splitResult = item[0].split('>')
            if len(splitResult) == 2:
                featureList.append(splitResult[0])
            elif len(splitResult) == 3:
                featureList.append(splitResult[1])
            else:
                raise ValueError("splitResult {} error!".format(splitResult))
        else:
            raise ValueError("splitResult {} error!".format(splitResult))

    for i in range(len(featureList)):
        if featureList[i][0] == ' ':
            featureList[i] = featureList[i][1:]

        if featureList[i][-1] == ' ':
            featureList[i] = featureList[i][:-1]

    return featureList

def batch_predict(data, model=lstmModel.model):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """

    # X_tensor = torch.from_numpy(data).float()
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    X_train = more_itertools.windowed(data, n=1, step=1)
    X_train = np.asarray(list(X_train))
    X_train = torch.from_numpy(X_train).type(torch.float)

    out, preOut, multiOut = model(X_train)
    probs = torch.nn.functional.softmax(preOut, dim=1)
    return probs.detach().cpu().numpy()

def batch_predict02(data, model=lstmModel02.model):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """

    # X_tensor = torch.from_numpy(data).float()
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    X_train = more_itertools.windowed(data, n=1, step=1)
    X_train = np.asarray(list(X_train))
    X_train = torch.from_numpy(X_train).type(torch.float)

    out, preOut, multiOut = model(X_train)
    probs = torch.nn.functional.softmax(preOut, dim=1)
    return probs.detach().cpu().numpy()

def batch_predict03(data, model=lstmModel03.model):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """

    # X_tensor = torch.from_numpy(data).float()
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    X_train = more_itertools.windowed(data, n=1, step=1)
    X_train = np.asarray(list(X_train))
    X_train = torch.from_numpy(X_train).type(torch.float)

    out, preOut, multiOut = model(X_train)
    probs = torch.nn.functional.softmax(preOut, dim=1)
    return probs.detach().cpu().numpy()

if __name__ == '__main__':
    expertLabelCount = 0
    normalLabelCount = 0
    preTrainData, preTrainLabel, trainData, trainLabel, testData, testLabel = lstmModel.inputData()
    deleteDataList = []
    deleteLabelList = []

    #################  预训练  #################
    lstmModel.trainModel(preTrainData, preTrainLabel, epoches=3000)
    state = {'net': lstmModel.model.state_dict(), 'optimizer': lstmModel.optimizier.state_dict(), 'epoch': 3000}
    torch.save(state, './model/lstm.pth')

    lstmModel02.trainModel(preTrainData, preTrainLabel, epoches=2000)
    state = {'net': lstmModel02.model.state_dict(), 'optimizer': lstmModel02.optimizier.state_dict(), 'epoch': 2000}
    torch.save(state, './model/lstm02.pth')

    lstmModel03.trainModel(preTrainData, preTrainLabel, epoches=2000)
    state = {'net': lstmModel03.model.state_dict(), 'optimizer': lstmModel03.optimizier.state_dict(), 'epoch': 2000}
    torch.save(state, './model/lstm03.pth')


    lstmModel.loadModel()
    lstmModel02.loadModel()
    lstmModel03.loadModel()

    # normer = Normalizer(preTrainData[:, 1:].shape[-1], online_minmax=False)
    # preTrainData_feat = normer.fit_transform(preTrainData[:, 1:])
    # preTrainData_feat = np.insert(preTrainData_feat, 0, values=preTrainData[:, 0], axis=1)
    dc.cluster(preTrainData, preTrainLabel)
    dc02.cluster(preTrainData, preTrainLabel)
    dc03.cluster(preTrainData, preTrainLabel)

    #################  训练循环  #################
    localStability = 0
    globalStability = 0
    ourStability = 0

    localAopc = 0
    globalAopc = 0
    ourAopc = 0

    localRobustness = 0
    globalRobustness = 0
    ourRobustness = 0

    t = 0
    loopCount = 0
    normer = Normalizer(trainData[:, 1:].shape[-1], online_minmax=False)
    while(list(trainLabel) != []):
        loopCount += 1
        if loopCount == 10:
            # 提取lstm中间层结果
            indexs, clusterSample = lstmModel.featureExtract(trainData)

            for i in range(len(trainData)):
                preTrainData = np.row_stack((preTrainData, trainData[i]))
                preTrainLabel = np.append(preTrainLabel, trainLabel[i])

                dc.clustervoteResultDict[trainLabel[i]].append(clusterSample[i])

            #################  重训练  #################
            lstmModel.trainModel(preTrainData, preTrainLabel, epoches=2000)

            break


        # 计算块数
        T = round(np.shape(trainData)[0] / batchSize)
        T = int(T)

        # 取出块数据
        if t == 0:
            sample = trainData[0:batchSize, :]
            labelStruck = trainLabel[0:batchSize]
            t += 1
        elif t >= T:
            t = 0
            continue
        else:
            sample = trainData[t * batchSize:(t + 1) * batchSize]
            labelStruck = trainLabel[t * batchSize:(t + 1) * batchSize]
            t += 1

        #################  获得预测值  #################

        train_feat = normer.fit_transform(sample[:, 1:])
        train_feat = np.insert(train_feat, 0, values=sample[:, 0], axis=1)

        # lstm预测
        indexs, lstmResult = lstmModel.predict(testData)

        # 提取lstm中间层结果
        indexs, clusterSample = lstmModel.featureExtract(trainData)

        # dc预测
        indexs, dcResult = dc.predict(testData)

        # 跟踪lstm与dc的效果
        print("lstm accuracy: ", accuracy_score(testLabel, lstmResult))
        print("dc accuracy: ", accuracy_score(testLabel, dcResult))

        # lstm预测
        indexs, lstmResult = lstmModel.predict(sample)

        # 提取lstm中间层结果
        indexs, clusterSample = lstmModel.featureExtract(sample)

        # dc预测
        indexs, dcResult = dc.predict(sample)
        indexs, dcResult02 = dc02.predict(sample)
        indexs, dcResult03 = dc03.predict(sample)

        # 跟踪lstm与dc的效果
        print("lstm accuracy: ", accuracy_score(labelStruck, lstmResult))
        print("dc accuracy: ", accuracy_score(labelStruck, dcResult))


        #################  解释器定义  #################

        # Local surrogate
        # localExplainer = LimeTabularExplainer(sample[:, 1:],
        localExplainer = LimeTabularExplainer(preTrainData[:, 1:],
                                         feature_names=features_names,
                                         class_names=targets,
                                         discretize_continuous=True, )

        # Global surrogate
        model_task = "classification"
        globalExplainer = PFIExplainer(dc, is_function=False, model_task=model_task, features=features_names)

        preTrainDataCopy = copy(preTrainData)
        deleteDataList = []
        deleteLabelList = []

        if len(sample) < batchSize:
            forNum = len(sample)
        else:
            forNum = batchSize
        for i in range(forNum):
            if dcResult[i] == -1:
                continue

            #################  提取重要特征  #################0
            tempSample = sample[:, 1:][i]

            # local surrogate 输出
            exp = localExplainer.explain_instance(tempSample,
                                                  batch_predict,
                                                  num_features=26,
                                                  num_samples=1280,
                                                  top_labels=8)

            localExplainerResult = sorted(exp.as_list(), key=lambda x: x[1], reverse=True)
            localFeature = localFeatureExtract(localExplainerResult)[:topFeatureNum]

            globalDataInput = []
            for globalI in dc.clustervoteIndexDict[dcResult[i]]:
                for globalJ in preTrainDataCopy:
                    if int(globalJ[0]) == globalI:
                        globalDataInput.append(copy(globalJ))
                        break


            # global surrogate 输出
            globalResult = globalExplainer.explain_global(np.asarray(globalDataInput)[:, 1:], true_labels=np.asarray([dc.clustervoteLabelDict[dcResult[i]] for j in range(len(dc.clustervoteIndexDict[dcResult[i]]))]))
            globalFeature = globalResult.get_ranked_global_names()[:topFeatureNum]

            # 计算similar score
            simResult = simCheck(localFeature, globalFeature)

            # #################  计算metric  #################
            # #################  计算stability  #################
            # 计算local stability
            ModelNum = 3
            compareNum = 2

            exp02 = localExplainer.explain_instance(tempSample,
                                                    batch_predict02,
                                                    num_features=26,
                                                    num_samples=1280,
                                                    top_labels=8)
            localExplainerResult02 = sorted(exp02.as_list(), key=lambda x: x[1], reverse=True)
            localFeature02 = localFeatureExtract(localExplainerResult02)[:topFeatureNum]

            exp03 = localExplainer.explain_instance(tempSample,
                                                    batch_predict03,
                                                    num_features=26,
                                                    num_samples=1280,
                                                    top_labels=8)
            localExplainerResult03 = sorted(exp03.as_list(), key=lambda x: x[1], reverse=True)
            localFeature03 = localFeatureExtract(localExplainerResult03)[:topFeatureNum]

            explist = [[localFeature, localFeature02], [localFeature, localFeature03],
                       [localFeature02, localFeature03]]
            simSum = 0
            for iexp in explist:
                simscore = simCheck(iexp[0], iexp[1])
                simSum += simscore


            def fact(i):
                return math.factorial(i)


            stab = fact(ModelNum) / (fact(compareNum) * fact(ModelNum - compareNum))
            stab = 1 / stab
            stab = stab * simSum
            localstab = stab
            localStability += stab

            # 计算global stability
            ModelNum = 3
            compareNum = 2

            # globalResult01 = globalExplainer.explain_global(np.asarray(tempSample).reshape(1, -1),
            #                                                 true_labels=np.asarray([dcResult[i]]).reshape(1, -1))

            globalResult01 = globalExplainer.explain_global(np.asarray(globalDataInput[:10])[:, 1:],
                                                            true_labels=np.asarray(
                                                                [dc.clustervoteLabelDict[dcResult[i]] for j in
                                                                 range(10)]))

            globalFeature01 = globalResult01.get_ranked_global_names()[:topFeatureNum]

            globalExplainer02 = PFIExplainer(dc02, is_function=False, model_task=model_task, features=features_names)
            # globalDataInput02 = []
            # for globalI in dc02.clustervoteIndexDict[dcResult02[i]]:
            #     for globalJ in preTrainDataCopy:
            #         if int(globalJ[0]) == globalI:
            #             globalDataInput02.append(copy(globalJ))
            #             break

            # global surrogate 输出
            # globalResult02 = globalExplainer02.explain_global(np.asarray(globalDataInput02)[:, 1:],
            #                                                   true_labels=np.asarray(
            #                                                       [dc02.clustervoteLabelDict[dcResult02[i]] for j in
            #                                                        range(
            #                                                            len(dc02.clustervoteIndexDict[dcResult02[i]]))]))

            # globalResult02 = globalExplainer02.explain_global(np.asarray(tempSample).reshape(1, -1),
            #                                                   true_labels=np.asarray([dcResult[i]]).reshape(1, -1))

            globalResult02 = globalExplainer02.explain_global(np.asarray(globalDataInput[:10])[:, 1:],
                                                              true_labels=np.asarray(
                                                                  [dc.clustervoteLabelDict[dcResult[i]] for j in
                                                                   range(10)]))
            globalFeature02 = globalResult02.get_ranked_global_names()[:topFeatureNum]

            globalExplainer03 = PFIExplainer(dc03, is_function=False, model_task=model_task, features=features_names)
            # globalDataInput03 = []
            # for globalI in dc03.clustervoteIndexDict[dcResult03[i]]:
            #     for globalJ in preTrainDataCopy:
            #         if int(globalJ[0]) == globalI:
            #             globalDataInput03.append(copy(globalJ))
            #             break

            # global surrogate 输出
            # globalResult03 = globalExplainer03.explain_global(np.asarray(globalDataInput03)[:, 1:],
            #                                                   true_labels=np.asarray(
            #                                                       [dc03.clustervoteLabelDict[dcResult03[i]] for j in
            #                                                        range(
            #                                                            len(dc03.clustervoteIndexDict[dcResult03[i]]))]))

            # globalResult03 = globalExplainer03.explain_global(np.asarray(tempSample).reshape(1, -1),
            # true_labels=np.asarray([dcResult[i]]).reshape(1, -1))
            globalResult03 = globalExplainer03.explain_global(np.asarray(globalDataInput[:10])[:, 1:],
                                                              true_labels=np.asarray(
                                                                  [dc.clustervoteLabelDict[dcResult[i]] for j in
                                                                   range(10)]))
            globalFeature03 = globalResult03.get_ranked_global_names()[:topFeatureNum]

            explist = [[globalFeature01, globalFeature02], [globalFeature01, globalFeature03],
                       [globalFeature02, globalFeature03]]
            simSum = 0
            for iexp in explist:
                simscore = simCheck(iexp[0], iexp[1])
                simSum += simscore


            def fact(i):
                return math.factorial(i)


            stab = fact(ModelNum) / (fact(compareNum) * fact(ModelNum - compareNum))
            stab = 1 / stab
            stab = stab * simSum
            globalstab = stab
            globalStability += stab

            # 计算our stability
            if localstab > globalstab:
                ourStability += localstab
            else:
                ourStability += globalstab

            #################  计算aopc  #################
            # 计算local aopc
            aopcSample = deepcopy(sample)
            aopcSum = 0

            lstmRawOuput = batch_predict(aopcSample[:, 1:])[i]
            maxIndex = np.argmax(lstmRawOuput)
            predictRaw = lstmRawOuput[maxIndex]

            localFeatureIndex = []
            for featureItem in localFeature:
                localFeatureIndex.append(features_names.index(featureItem))

            for featureIndex in localFeatureIndex:
                aopcSample[:, featureIndex] = 0
                lstmTempOuput = batch_predict(aopcSample[:, 1:])[i]
                predictTemp = lstmTempOuput[maxIndex]
                aopcSum += abs(predictRaw - predictTemp)
            localAopcSum = aopcSum / (topFeatureNum + 1)

            localAopc += aopcSum / (topFeatureNum + 1)

            # 计算global aopc
            aopcSample = deepcopy(sample)
            aopcSum = 0

            dcRawOuput = dc(torch.from_numpy(aopcSample[:, 1:]).type(torch.float)).numpy()[i]
            maxIndex = np.argmax(dcRawOuput)
            predictRaw = dcRawOuput[maxIndex]

            globalFeatureIndex = []
            for featureItem in globalFeature:
                globalFeatureIndex.append(features_names.index(featureItem))

            for featureIndex in globalFeatureIndex:
                aopcSample[:, featureIndex] = 0
                dcTempOuput = batch_predict(aopcSample[:, 1:])[i]
                predictTemp = dcTempOuput[maxIndex]
                aopcSum += abs(predictRaw - predictTemp)
            globalAopcSum = aopcSum / (topFeatureNum + 1)

            globalAopc += aopcSum / (topFeatureNum + 1)

            # 计算our aopc
            if localAopcSum > globalAopcSum:
                ourAopc += localAopcSum
            else:
                ourAopc += globalAopcSum

            #################  计算robustness  #################
            # 计算local robustness
            predictResult = lstmResult[i]
            localRSum = 0
            localWSum = 0
            countR = 0
            countW = 0

            for localRItem in range(len(lstmResult)):
                if lstmResult[localRItem] == predictResult:
                    # local surrogate 输出
                    expLocalR = localExplainer.explain_instance(sample[:, 1:][localRItem],
                                                                batch_predict,
                                                                num_features=26,
                                                                num_samples=1280,
                                                                top_labels=8)

                    localExplainerResultLocalR = sorted(expLocalR.as_list(), key=lambda x: x[1], reverse=True)
                    localFeatureLocalR = localFeatureExtract(localExplainerResultLocalR)[:topFeatureNum]
                    resultLocalR = simCheck(localFeature, localFeatureLocalR)
                    localRSum += resultLocalR
                    countR += 1
                else:
                    # local surrogate 输出
                    expLocalW = localExplainer.explain_instance(sample[:, 1:][localRItem],
                                                                batch_predict,
                                                                num_features=26,
                                                                num_samples=1280,
                                                                top_labels=8)

                    localExplainerResultLocalW = sorted(expLocalW.as_list(), key=lambda x: x[1], reverse=True)
                    localFeatureLocalW = localFeatureExtract(localExplainerResultLocalW)[:topFeatureNum]
                    resultLocalW = simCheck(localFeature, localFeatureLocalW)
                    localWSum += resultLocalW
                    countW += 1
            if countR != 0 and countW != 0:
                localRobustness += abs((localRSum / countR) - (localWSum / countW))
                localRobustnessTemp = abs((localRSum / countR) - (localWSum / countW))
            elif countR != 0 and countW == 0:
                localRobustness += abs((localRSum / countR))
                localRobustnessTemp = abs((localRSum / countR))
            else:
                localRobustness += abs((localWSum / countW))
                localRobustnessTemp = abs((localWSum / countW))

            # 计算global robustness
            predictResultGlobal = dcResult[i]
            globalRSum = 0
            globalWSum = 0
            countRGlobal = 0
            countWGlobal = 0

            for globalRItem in range(len(dcResult)):
                if dcResult[globalRItem] == predictResultGlobal:
                    # global surrogate 输出
                    globalDataInputR = []
                    for globalIR in dc.clustervoteIndexDict[dcResult[globalRItem]]:
                        for globalJW in preTrainDataCopy:
                            if int(globalJW[0]) == globalIR:
                                globalDataInputR.append(copy(globalJW))
                                break

                    tttInputR = list(globalDataInputR[:10])
                    tttInputR.append(sample[globalRItem])

                    # globalResultR = globalExplainer.explain_global(np.asarray(sample[globalRItem][1:]).reshape(1, -1),
                    #                                                true_labels=np.asarray(
                    #                                                    [dcResult[globalRItem]]).reshape(1, -1))
                    globalResultR = globalExplainer.explain_global(np.asarray(tttInputR)[:, 1:],
                                                                   true_labels=np.asarray(
                                                                       [dc.clustervoteLabelDict[dcResult[globalRItem]]
                                                                        for j in
                                                                        range(11)]))
                    globalFeatureR = globalResultR.get_ranked_global_names()[:topFeatureNum]
                    resultGlobalR = simCheck(globalFeature, globalFeatureR)
                    globalRSum += resultGlobalR
                    countRGlobal += 1
                else:
                    globalDataInputW = []
                    for globalIW in dc.clustervoteIndexDict[dcResult[globalRItem]]:
                        for globalJW in preTrainDataCopy:
                            if int(globalJW[0]) == globalIW:
                                globalDataInputW.append(copy(globalJW))
                                break

                    # globalDataInputW.append(sample[globalRItem])

                    tttInputW = list(globalDataInputR[:10])
                    tttInputW.append(sample[globalRItem])

                    # global surrogate 输出
                    # globalResultW = globalExplainer.explain_global(np.asarray(sample[globalRItem][1:]).reshape(1, -1),
                    #                                                true_labels=np.asarray(
                    #                                                    [dcResult[globalRItem]]).reshape(1, -1))
                    globalResultW = globalExplainer.explain_global(np.asarray(tttInputW)[:, 1:],
                                                                   true_labels=np.asarray(
                                                                       [dc.clustervoteLabelDict[dcResult[globalRItem]]
                                                                        for j in
                                                                        range(11)]))
                    globalFeatureW = globalResultW.get_ranked_global_names()[:topFeatureNum]
                    resultGlobalW = simCheck(globalFeature, globalFeatureW)
                    globalWSum += resultGlobalW
                    countWGlobal += 1

            globalRobustnessTemp = 0
            if countRGlobal != 0 and countWGlobal != 0:
                globalRobustness += abs((globalRSum / countRGlobal) - (globalWSum / countWGlobal))
                globalRobustnessTemp = abs((globalRSum / countRGlobal) - (globalWSum / countWGlobal))
            elif countRGlobal != 0 and countWGlobal == 0:
                globalRobustness += abs((globalRSum / countRGlobal))
                globalRobustnessTemp = abs((globalRSum / countRGlobal))
            else:
                globalRobustness += abs((globalWSum / countWGlobal))
                globalRobustnessTemp = abs((globalWSum / countWGlobal))

            # 计算our robustness
            if localRobustnessTemp > globalRobustnessTemp:
                ourRobustness += localRobustnessTemp
            else:
                ourRobustness += globalRobustnessTemp


    #################  测试循环  #################
    localStability = 0
    globalStability = 0
    ourStability = 0

    localAopc = 0
    globalAopc = 0
    ourAopc = 0

    localRobustness = 0
    globalRobustness = 0
    ourRobustness = 0

    t = 0
    resultTruth = []
    resultTest = []
    loopCount = 0
    normer = Normalizer(testData[:, 1:].shape[-1], online_minmax=False)
    while (list(testLabel) != []):
        loopCount += 1
        if loopCount == 10:
            # 提取lstm中间层结果
            indexs, clusterSample = lstmModel.featureExtract(testData)

            for i in range(len(testData)):
                preTrainData = np.row_stack((preTrainData, testData[i]))
                preTrainLabel = np.append(preTrainLabel, testLabel[i])

                dc.clustervoteResultDict[testLabel[i]].append(clusterSample[i])

                resultTruth.append(testLabel[i])
                resultTest.append(testLabel[i])

                expertLabelCount += 1

            #################  重训练  #################
            lstmModel.trainModel(preTrainData, preTrainLabel, epoches=500)

            break

        # 计算块数
        T = round(np.shape(testData)[0] / batchSize)
        T = int(T)

        # 取出块数据
        if t == 0:
            sample = testData[0:batchSize, :]
            labelStruck = testLabel[0:batchSize]
            t += 1
        elif t >= T:
            t = 0
            continue
        else:
            sample = testData[t * batchSize:(t + 1) * batchSize]
            labelStruck = testLabel[t * batchSize:(t + 1) * batchSize]
            t += 1

        #################  获得预测值  #################

        test_feat = normer.fit_transform(sample[:, 1:])
        test_feat = np.insert(test_feat, 0, values=sample[:, 0], axis=1)

        # lstm预测
        indexs, lstmResult = lstmModel.predict(sample)

        # 提取lstm中间层结果
        indexs, clusterSample = lstmModel.featureExtract(sample)

        # dc预测
        indexs, dcResult = dc.predict(sample)
        indexs, dcResult02 = dc02.predict(sample)
        indexs, dcResult03 = dc03.predict(sample)

        #################  解释器定义  #################

        # Local surrogate
        # localExplainer = LimeTabularExplainer(sample[:, 1:],
        localExplainer = LimeTabularExplainer(preTrainData[:, 1:],
                                              feature_names=features_names,
                                              class_names=targets,
                                              discretize_continuous=True, )

        # Global surrogate
        model_task = "classification"
        globalExplainer = PFIExplainer(dc, is_function=False, model_task=model_task, features=features_names)

        sampleCopy = deepcopy(sample)
        deleteDataList = []
        deleteLabelList = []

        if len(sampleCopy) < batchSize:
            forNum = len(sampleCopy)
        else:
            forNum = batchSize
        for i in range(forNum):
            if dcResult[i] == -1:
                continue

            #################  提取重要特征  #################
            tempSample = sample[:, 1:][i]

            # local surrogate 输出
            exp = localExplainer.explain_instance(tempSample,
                                                  batch_predict,
                                                  num_features=26,
                                                  num_samples=1280,
                                                  top_labels=8)

            localExplainerResult = sorted(exp.as_list(), key=lambda x: x[1], reverse=True)
            localFeature = localFeatureExtract(localExplainerResult)[:topFeatureNum]

            globalDataInput = []
            for globalI in dc.clustervoteIndexDict[dcResult[i]]:
                for globalJ in preTrainDataCopy:
                    if int(globalJ[0]) == globalI:
                        globalDataInput.append(copy(globalJ))
                        break

            # global surrogate 输出
            globalResult = globalExplainer.explain_global(np.asarray(globalDataInput)[:, 1:], true_labels=np.asarray(
                [dc.clustervoteLabelDict[dcResult[i]] for j in range(len(dc.clustervoteIndexDict[dcResult[i]]))]))
            globalFeature = globalResult.get_ranked_global_names()[:topFeatureNum]

            # 计算similar score
            simResult = simCheck(localFeature, globalFeature)

            # # #################  计算metric  #################
            # # #################  计算stability  #################
            # # 计算local stability
            # ModelNum = 3
            # compareNum = 2
            #
            # exp02 = localExplainer.explain_instance(tempSample,
            #                                         batch_predict02,
            #                                         num_features=26,
            #                                         num_samples=1280,
            #                                         top_labels=8)
            # localExplainerResult02 = sorted(exp02.as_list(), key=lambda x: x[1], reverse=True)
            # localFeature02 = localFeatureExtract(localExplainerResult02)[:topFeatureNum]
            #
            # exp03 = localExplainer.explain_instance(tempSample,
            #                                         batch_predict03,
            #                                         num_features=26,
            #                                         num_samples=1280,
            #                                         top_labels=8)
            # localExplainerResult03 = sorted(exp03.as_list(), key=lambda x: x[1], reverse=True)
            # localFeature03 = localFeatureExtract(localExplainerResult03)[:topFeatureNum]
            #
            # explist = [[localFeature, localFeature02], [localFeature, localFeature03],
            #            [localFeature02, localFeature03]]
            # simSum = 0
            # for iexp in explist:
            #     simscore = simCheck(iexp[0], iexp[1])
            #     simSum += simscore
            #
            #
            # def fact(i):
            #     return math.factorial(i)
            #
            #
            # stab = fact(ModelNum) / (fact(compareNum) * fact(ModelNum - compareNum))
            # stab = 1 / stab
            # stab = stab * simSum
            # localstab = stab
            # localStability += stab
            #
            # # 计算global stability
            # ModelNum = 3
            # compareNum = 2
            #
            # # globalResult01 = globalExplainer.explain_global(np.asarray(tempSample).reshape(1, -1),
            # #                                                 true_labels=np.asarray([dcResult[i]]).reshape(1, -1))
            #
            # globalResult01 = globalExplainer.explain_global(np.asarray(globalDataInput[:10])[:, 1:],
            #                                                 true_labels=np.asarray(
            #                                                     [dc.clustervoteLabelDict[dcResult[i]] for j in
            #                                                      range(10)]))
            #
            # globalFeature01 = globalResult01.get_ranked_global_names()[:topFeatureNum]
            #
            # globalExplainer02 = PFIExplainer(dc02, is_function=False, model_task=model_task, features=features_names)
            # # globalDataInput02 = []
            # # for globalI in dc02.clustervoteIndexDict[dcResult02[i]]:
            # #     for globalJ in preTrainDataCopy:
            # #         if int(globalJ[0]) == globalI:
            # #             globalDataInput02.append(copy(globalJ))
            # #             break
            #
            # # global surrogate 输出
            # # globalResult02 = globalExplainer02.explain_global(np.asarray(globalDataInput02)[:, 1:],
            # #                                                   true_labels=np.asarray(
            # #                                                       [dc02.clustervoteLabelDict[dcResult02[i]] for j in
            # #                                                        range(
            # #                                                            len(dc02.clustervoteIndexDict[dcResult02[i]]))]))
            #
            # # globalResult02 = globalExplainer02.explain_global(np.asarray(tempSample).reshape(1, -1),
            # #                                                   true_labels=np.asarray([dcResult[i]]).reshape(1, -1))
            #
            # globalResult02 = globalExplainer02.explain_global(np.asarray(globalDataInput[:10])[:, 1:],
            #                                                 true_labels=np.asarray(
            #                                                     [dc.clustervoteLabelDict[dcResult[i]] for j in
            #                                                      range(10)]))
            # globalFeature02 = globalResult02.get_ranked_global_names()[:topFeatureNum]
            #
            # globalExplainer03 = PFIExplainer(dc03, is_function=False, model_task=model_task, features=features_names)
            # # globalDataInput03 = []
            # # for globalI in dc03.clustervoteIndexDict[dcResult03[i]]:
            # #     for globalJ in preTrainDataCopy:
            # #         if int(globalJ[0]) == globalI:
            # #             globalDataInput03.append(copy(globalJ))
            # #             break
            #
            # # global surrogate 输出
            # # globalResult03 = globalExplainer03.explain_global(np.asarray(globalDataInput03)[:, 1:],
            # #                                                   true_labels=np.asarray(
            # #                                                       [dc03.clustervoteLabelDict[dcResult03[i]] for j in
            # #                                                        range(
            # #                                                            len(dc03.clustervoteIndexDict[dcResult03[i]]))]))
            #
            # # globalResult03 = globalExplainer03.explain_global(np.asarray(tempSample).reshape(1, -1),
            #                                                   # true_labels=np.asarray([dcResult[i]]).reshape(1, -1))
            # globalResult03 = globalExplainer03.explain_global(np.asarray(globalDataInput[:10])[:, 1:],
            #                                                 true_labels=np.asarray(
            #                                                     [dc.clustervoteLabelDict[dcResult[i]] for j in
            #                                                      range(10)]))
            # globalFeature03 = globalResult03.get_ranked_global_names()[:topFeatureNum]
            #
            # explist = [[globalFeature01, globalFeature02], [globalFeature01, globalFeature03],
            #            [globalFeature02, globalFeature03]]
            # simSum = 0
            # for iexp in explist:
            #     simscore = simCheck(iexp[0], iexp[1])
            #     simSum += simscore
            #
            #
            # def fact(i):
            #     return math.factorial(i)
            #
            #
            # stab = fact(ModelNum) / (fact(compareNum) * fact(ModelNum - compareNum))
            # stab = 1 / stab
            # stab = stab * simSum
            # globalstab = stab
            # globalStability += stab
            #
            # # 计算our stability
            # if localstab > globalstab:
            #     ourStability += localstab
            # else:
            #     ourStability += globalstab
            #
            # #################  计算aopc  #################
            # # 计算local aopc
            # aopcSample = deepcopy(sample)
            # aopcSum = 0
            #
            # lstmRawOuput = batch_predict(aopcSample[:, 1:])[i]
            # maxIndex = np.argmax(lstmRawOuput)
            # predictRaw = lstmRawOuput[maxIndex]
            #
            # localFeatureIndex = []
            # for featureItem in localFeature:
            #     localFeatureIndex.append(features_names.index(featureItem))
            #
            # for featureIndex in localFeatureIndex:
            #     aopcSample[:, featureIndex] = 0
            #     lstmTempOuput = batch_predict(aopcSample[:, 1:])[i]
            #     predictTemp = lstmTempOuput[maxIndex]
            #     aopcSum += abs(predictRaw - predictTemp)
            # localAopcSum = aopcSum / (topFeatureNum + 1)
            #
            # localAopc += aopcSum / (topFeatureNum + 1)
            #
            # # 计算global aopc
            # aopcSample = deepcopy(sample)
            # aopcSum = 0
            #
            # dcRawOuput = dc(torch.from_numpy(aopcSample[:, 1:]).type(torch.float)).numpy()[i]
            # maxIndex = np.argmax(dcRawOuput)
            # predictRaw = dcRawOuput[maxIndex]
            #
            # globalFeatureIndex = []
            # for featureItem in globalFeature:
            #     globalFeatureIndex.append(features_names.index(featureItem))
            #
            # for featureIndex in globalFeatureIndex:
            #     aopcSample[:, featureIndex] = 0
            #     dcTempOuput = batch_predict(aopcSample[:, 1:])[i]
            #     predictTemp = dcTempOuput[maxIndex]
            #     aopcSum += abs(predictRaw - predictTemp)
            # globalAopcSum = aopcSum / (topFeatureNum + 1)
            #
            # globalAopc += aopcSum / (topFeatureNum + 1)
            #
            # # 计算our aopc
            # if localAopcSum > globalAopcSum:
            #     ourAopc += localAopcSum
            # else:
            #     ourAopc += globalAopcSum
            #
            # #################  计算robustness  #################
            # # 计算local robustness
            # predictResult = lstmResult[i]
            # localRSum = 0
            # localWSum = 0
            # countR = 0
            # countW = 0
            #
            # for localRItem in range(len(lstmResult)):
            #     if lstmResult[localRItem] == predictResult:
            #         # local surrogate 输出
            #         expLocalR = localExplainer.explain_instance(sample[:, 1:][localRItem],
            #                                                     batch_predict,
            #                                                     num_features=26,
            #                                                     num_samples=1280,
            #                                                     top_labels=8)
            #
            #         localExplainerResultLocalR = sorted(expLocalR.as_list(), key=lambda x: x[1], reverse=True)
            #         localFeatureLocalR = localFeatureExtract(localExplainerResultLocalR)[:topFeatureNum]
            #         resultLocalR = simCheck(localFeature, localFeatureLocalR)
            #         localRSum += resultLocalR
            #         countR += 1
            #     else:
            #         # local surrogate 输出
            #         expLocalW = localExplainer.explain_instance(sample[:, 1:][localRItem],
            #                                                     batch_predict,
            #                                                     num_features=26,
            #                                                     num_samples=1280,
            #                                                     top_labels=8)
            #
            #         localExplainerResultLocalW = sorted(expLocalW.as_list(), key=lambda x: x[1], reverse=True)
            #         localFeatureLocalW = localFeatureExtract(localExplainerResultLocalW)[:topFeatureNum]
            #         resultLocalW = simCheck(localFeature, localFeatureLocalW)
            #         localWSum += resultLocalW
            #         countW += 1
            # if countR != 0 and countW != 0:
            #     localRobustness += abs((localRSum / countR) - (localWSum / countW))
            #     localRobustnessTemp = abs((localRSum / countR) - (localWSum / countW))
            # elif countR != 0 and countW == 0:
            #     localRobustness += abs((localRSum / countR))
            #     localRobustnessTemp = abs((localRSum / countR))
            # else:
            #     localRobustness += abs((localWSum / countW))
            #     localRobustnessTemp = abs((localWSum / countW))
            #
            # # 计算global robustness
            # predictResultGlobal = dcResult[i]
            # globalRSum = 0
            # globalWSum = 0
            # countRGlobal = 0
            # countWGlobal = 0
            #
            # for globalRItem in range(len(dcResult)):
            #     if dcResult[globalRItem] == predictResultGlobal:
            #         # global surrogate 输出
            #         globalDataInputR = []
            #         for globalIR in dc.clustervoteIndexDict[dcResult[globalRItem]]:
            #             for globalJW in preTrainDataCopy:
            #                 if int(globalJW[0]) == globalIR:
            #                     globalDataInputR.append(copy(globalJW))
            #                     break
            #
            #         globalDataInputR.append(sample[globalRItem])
            #
            #         # globalResultR = globalExplainer.explain_global(np.asarray(sample[globalRItem][1:]).reshape(1, -1),
            #         #                                                true_labels=np.asarray(
            #         #                                                    [dcResult[globalRItem]]).reshape(1, -1))
            #         globalResultR = globalExplainer.explain_global(np.asarray(globalDataInputR[:10])[:, 1:],
            #                                                         true_labels=np.asarray(
            #                                                             [dc.clustervoteLabelDict[dcResult[globalRItem]] for j in
            #                                                              range(10)]))
            #         globalFeatureR = globalResultR.get_ranked_global_names()[:topFeatureNum]
            #         resultGlobalR = simCheck(globalFeature, globalFeatureR)
            #         globalRSum += resultGlobalR
            #         countRGlobal += 1
            #     else:
            #         globalDataInputW = []
            #         for globalIW in dc.clustervoteIndexDict[dcResult[globalRItem]]:
            #             for globalJW in preTrainDataCopy:
            #                 if int(globalJW[0]) == globalIW:
            #                     globalDataInputW.append(copy(globalJW))
            #                     break
            #
            #         globalDataInputW.append(sample[globalRItem])
            #
            #         # global surrogate 输出
            #         # globalResultW = globalExplainer.explain_global(np.asarray(sample[globalRItem][1:]).reshape(1, -1),
            #         #                                                true_labels=np.asarray(
            #         #                                                    [dcResult[globalRItem]]).reshape(1, -1))
            #         globalResultW = globalExplainer.explain_global(np.asarray(globalDataInputW[:10])[:, 1:],
            #                                                        true_labels=np.asarray(
            #                                                            [dc.clustervoteLabelDict[dcResult[globalRItem]]
            #                                                             for j in
            #                                                             range(10)]))
            #         globalFeatureW = globalResultW.get_ranked_global_names()[:topFeatureNum]
            #         resultGlobalW = simCheck(globalFeature, globalFeatureW)
            #         globalWSum += resultGlobalW
            #         countWGlobal += 1
            #
            # globalRobustnessTemp = 0
            # if countRGlobal != 0 and countWGlobal != 0:
            #     globalRobustness += abs((globalRSum / countRGlobal) - (globalWSum / countWGlobal))
            #     globalRobustnessTemp = abs((globalRSum / countRGlobal) - (globalWSum / countWGlobal))
            # elif countRGlobal != 0 and countWGlobal == 0:
            #     globalRobustness += abs((globalRSum / countRGlobal))
            #     globalRobustnessTemp = abs((globalRSum / countRGlobal))
            # else:
            #     globalRobustness += abs((globalWSum / countWGlobal))
            #     globalRobustnessTemp = abs((globalWSum / countWGlobal))
            #
            # # 计算our robustness
            # if localRobustnessTemp > globalRobustnessTemp:
            #     ourRobustness += localRobustnessTemp
            # else:
            #     ourRobustness += globalRobustnessTemp
