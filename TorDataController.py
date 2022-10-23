import copy
import random

import numpy as np
import csv

import sklearn
from sklearn.model_selection import train_test_split

from utils import Normalizer
from collections import Counter

labelDict = {
    'AUDIO': '0',
    'BROWSING': '1',
    'CHAT': '2',
    'FILE-TRANSFER': '3',
    'MAIL': '4',
    'VIDEO': '5',
    'VOIP': '6',
    'P2P': '7',
}

dictLabel = {
    '0': 'AUDIO',
    '1': 'BROWSING',
    '2': 'CHAT',
    '3': 'FILE-TRANSFER',
    '4': 'MAIL',
    '5': 'VIDEO',
    '6': 'VOIP',
    '7': 'P2P'
}

dataNum = {
        'AUDIO': 721,
        'BROWSING': 1604,
        'CHAT': 323,
        'FILE-TRANSFER': 864,
        'MAIL': 282,
        'VIDEO': 874,
        'VOIP': 2291,
        'P2P': 1085,
    }

fileList = [
    'csv\Tor.csv',
]

def handleNanInf(npdata):
    """
    删掉包含nan或inf的行
    :param npdata:
    :return:
    """
    nanList = ~np.isnan(npdata).any(axis=1)
    npdata = npdata[nanList]

    infList = ~np.isinf(npdata).any(axis=1)
    npdata = npdata[infList]

    return npdata

def NovelBuildRFAllAndSplit(csvFile='./csv/Tor.csv', sNum=200, selectLabel=[0,1,2,3,4,5,6,7]):

    labelSum = {
        '0': 721,
        '1': 1604,
        '2': 323,
        '3': 864,
        '4': 282,
        '5': 874,
        '6': 2291,
        '7': 1085,
    }

    selectSum = {

    }
    selectCount = {

    }

    pretainData = []
    tainData = []
    testData = []
    allData = []


    for item in range(0, 8):
        if item in selectLabel:
            selectSum[str(item)] = sNum
            selectCount[str(item)] = 0
        else:
            selectSum[str(item)] = 0
            selectCount[str(item)] = 0

    for filename in fileList:
        with open(filename, 'r') as fp:
            reader = csv.reader(fp)
            # index = 0
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                elif row[-1] in labelDict.keys():
                    data = row[:-1]
                    data.pop(0)
                    data.pop(1)
                    # data.pop(2)
                    # data.pop(18)
                    # data.pop(22)
                    data.append(labelDict[row[-1]])
                    # data.insert(0, str(index))
                    allData.append(data.copy())
                    # index += 1

    labelCol = copy.deepcopy(np.asarray(allData)[:, -1])
    allData = np.asarray(allData, dtype=float)
    normer = Normalizer(allData[:, :-1].shape[-1], online_minmax=False)
    allData = normer.fit_transform(allData[:, :-1])
    allData = np.insert(allData, allData.shape[1], values=labelCol, axis=1)
    allData = allData.tolist()

    for item in allData:
        item[-1] = int(item[-1])

    index = 0
    poisonInex = random.sample([i for i in range(158*8)], 158*5)
    for item in allData:
        if selectCount[str(int(item[-1]))] < 60:
            data = item[:]
            data.insert(0, str(index))
            # if index in poisonInex:
            #     data[len(data) - 1] = ((int(data[len(data) - 1]) + 3) % 8)
            pretainData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue
        elif selectCount[str(int(item[-1]))] < 160:
            data = item[:]
            data.insert(0, str(index))
            tainData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue
        elif selectCount[str(int(item[-1]))] < 200:
            data = item[:]
            data.insert(0, str(index))
            testData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue


    pretainData = np.asarray(pretainData)
    tainData = np.asarray(tainData)
    testData = np.asarray(testData)

    # np.random.shuffle(trainData)

    np.save('./Tor-novel/pretrain.npy', pretainData)
    np.save('./Tor-novel/train.npy', tainData)
    np.save('./Tor-novel/test.npy', testData)

    return pretainData, tainData, testData

def NovelBuildAllAndSplit(csvFile='./csv/Tor.csv', rate=0.2, rate2=0.7, selectLabel=[0,1,2,3,4,5,6,7]):

    labelSum = {
        '0': 721,
        '1': 1604,
        '2': 323,
        '3': 864,
        '4': 282,
        '5': 874,
        '6': 2291,
        '7': 1085,
    }

    selectSum = {

    }
    selectCount = {

    }

    selectSum2 = {

    }
    selectCount2 = {

    }

    pretainData = []
    trainData = []
    trainData2 = []
    testData = []
    allData = []

    for item in range(0, 8):
        if item in selectLabel:
            selectSum[str(item)] = int(labelSum[str(item)] * rate)
            selectCount[str(item)] = 0
            selectSum2[str(item)] = int(labelSum[str(item)] * rate2)
            selectCount2[str(item)] = 0
        else:
            selectSum[str(item)] = 0
            selectCount[str(item)] = 0
            selectSum2[str(item)] = 0
            selectCount2[str(item)] = 0

    for filename in fileList:
        with open(filename, 'r') as fp:
            reader = csv.reader(fp)
            # index = 0
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                elif row[-1] in labelDict.keys():
                    data = row[:-1]
                    data.pop(0)
                    data.pop(1)
                    # data.pop(2)
                    # data.pop(18)
                    # data.pop(22)
                    data.append(labelDict[row[-1]])
                    # data.insert(0, str(index))
                    allData.append(data.copy())
                    # index += 1

    labelCol = copy.deepcopy(np.asarray(allData)[:, -1])
    allData = np.asarray(allData, dtype=float)
    normer = Normalizer(allData[:, :-1].shape[-1], online_minmax=False)
    allData = normer.fit_transform(allData[:, :-1])
    allData = np.insert(allData, allData.shape[1], values=labelCol, axis=1)
    allData = allData.tolist()

    for item in allData:
        item[-1] = int(item[-1])

    index = 0
    poisonInex = random.sample([i for i in range(158*8)], 158*5)

    for item in allData:
        if selectCount[str(int(item[-1]))] < selectSum[str(int(item[-1]))]:
            data = item[:]
            data.insert(0, str(index))
            testData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue

        data = item[:]
        data.insert(0, str(index))
        trainData2.append((data.copy()))
        index += 1

    for item in trainData2:
        if selectCount2[str(int(item[-1]))] < selectSum2[str(int(item[-1]))]:
            data = item[:]
            pretainData.append(data.copy())
            selectCount2[str(int(item[-1]))] += 1
            continue

        data = item[:]
        trainData.append((data.copy()))


    pretainData = np.asarray(pretainData)
    trainData = np.asarray(trainData)
    testData = np.asarray(testData)

    # np.random.shuffle(trainData)

    np.save('./Tor-novel/pretrain.npy', pretainData)
    np.save('./Tor-novel/train.npy', trainData)
    np.save('./Tor-novel/test.npy', testData)

    return pretainData, trainData, testData

def NovelBuildRFAllAndSplitShuffle(csvFile='./csv/Tor.csv', sNum=200, selectLabel=[0,1,2,3,4,5,6,7]):

    labelSum = {
        '0': 721,
        '1': 1604,
        '2': 323,
        '3': 864,
        '4': 282,
        '5': 874,
        '6': 2291,
        '7': 1085,
    }

    selectSum = {

    }
    selectCount = {

    }

    pretainData = []
    tainData = []
    testData = []
    allData = []


    for item in range(0, 8):
        if item in selectLabel:
            selectSum[str(item)] = sNum
            selectCount[str(item)] = 0
        else:
            selectSum[str(item)] = 0
            selectCount[str(item)] = 0

    for filename in fileList:
        with open(filename, 'r') as fp:
            reader = csv.reader(fp)
            # index = 0
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                elif row[-1] in labelDict.keys():
                    data = row[:-1]
                    data.pop(0)
                    data.pop(1)
                    # data.pop(2)
                    # data.pop(18)
                    # data.pop(22)
                    data.append(labelDict[row[-1]])
                    # data.insert(0, str(index))
                    allData.append(data.copy())
                    # index += 1

    labelCol = copy.deepcopy(np.asarray(allData)[:, -1])
    allData = np.asarray(allData, dtype=float)
    normer = Normalizer(allData[:, :-1].shape[-1], online_minmax=False)
    allData = normer.fit_transform(allData[:, :-1])
    allData = np.insert(allData, allData.shape[1], values=labelCol, axis=1)
    allData = allData.tolist()

    random.shuffle(allData)

    for item in allData:
        item[-1] = int(item[-1])

    index = 0
    poisonInex = random.sample([i for i in range(158*8)], 158*5)
    for item in allData:
        if selectCount[str(int(item[-1]))] < 60:
            data = item[:]
            data.insert(0, str(index))
            # if index in poisonInex:
            #     data[len(data) - 1] = ((int(data[len(data) - 1]) + 3) % 8)
            pretainData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue
        elif selectCount[str(int(item[-1]))] < 160:
            data = item[:]
            data.insert(0, str(index))
            tainData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue
        elif selectCount[str(int(item[-1]))] < 200:
            data = item[:]
            data.insert(0, str(index))
            testData.append(data.copy())
            selectCount[str(int(item[-1]))] += 1
            index += 1
            continue


    pretainData = np.asarray(pretainData)
    tainData = np.asarray(tainData)
    testData = np.asarray(testData)

    # np.random.shuffle(trainData)

    np.save('./Tor-novel/pretrain.npy', pretainData)
    np.save('./Tor-novel/train.npy', tainData)
    np.save('./Tor-novel/test.npy', testData)

    return pretainData, tainData, testData

def TorNonTorCon(filename='./csv/TorNonTor.csv'):
    labelDict = {
        'nonTOR': 0,
        'TOR': 1
    }

    selectCount = {
        '0': 0,
        '1': 0
    }

    allData = []
    pretainData = []
    trainData = []
    testData = []

    with open(filename, 'r') as fp:
        reader = csv.reader(fp)
        # index = 0
        for i, row in enumerate(reader):
            if i == 0:
                continue
            elif row[-1] in labelDict.keys():
                data = row[:-1]
                data.pop(0)
                data.pop(1)
                # data.pop(2)
                # data.pop(18)
                # data.pop(22)
                data.append(labelDict[row[-1]])
                # data.insert(0, str(index))
                allData.append(data.copy())
                # index += 1

        index = 0
        poisonInex = random.sample([i for i in range(200 * 2, 250 * 2)], int(((250 - 200) * 1.0 * 2)))
        for item in allData:
            if selectCount[str(int(item[-1]))] < 75:
                data = item[:]
                data.insert(0, str(index))
                # if index in poisonInex:
                #     data[len(data) - 1] = ((int(data[len(data) - 1]) + 3) % 8)
                pretainData.append(data.copy())
                selectCount[str(int(item[-1]))] += 1
                index += 1
                continue
            elif selectCount[str(int(item[-1]))] < 200:
                data = item[:]
                data.insert(0, str(index))
                # if index in poisonInex:
                #     raC = random.random()
                #     if raC > 0.5:
                #         data[len(data) - 1] = ((int(data[len(data) - 1]) + 1) % 2)
                #     else:
                #         data[random.randint(1, 26)] = random.random()
                trainData.append(data.copy())
                selectCount[str(int(item[-1]))] += 1
                index += 1
                continue
            elif selectCount[str(int(item[-1]))] < 250:
                data = item[:]
                data.insert(0, str(index))
                if index in poisonInex:
                    data[random.randint(1, 26)] = random.random()
                    data[random.randint(1, 26)] = random.random() * 10
                    data[random.randint(1, 26)] = random.random() * 50
                    data[random.randint(1, 26)] = random.random() * 100
                testData.append(data.copy())
                selectCount[str(int(item[-1]))] += 1
                index += 1
                continue

        pretainData = np.asarray(pretainData)
        tainData = np.asarray(trainData)
        testData = np.asarray(testData)

        # np.random.shuffle(trainData)

        np.save('./Tor-NonTor/pretrain.npy', pretainData)
        np.save('./Tor-NonTor/train.npy', tainData)
        np.save('./Tor-NonTor/test.npy', testData)

        return pretainData, tainData, testData

def DarkNetCon(filename='./csv/Darknet.csv'):
    labelDict = {
        'Non-Tor': 1,
        'NonVPN': 0,
        'VPN': 1,
        'Tor': 1
    }

    selectCount = {
        '0': 0,
        '1': 0,
    }

    allData = []
    pretainData = []
    trainData = []
    testData = []

    with open(filename, 'r') as fp:
        reader = csv.reader(fp)
        # index = 0
        for i, row in enumerate(reader):
            if i == 0:
                continue
            elif row[-1] in labelDict.keys():
                data = row[:-1]
                data.pop(0)
                data.pop(0)
                data.pop(1)
                data.pop(3)
                # data.pop(2)
                # data.pop(18)
                # data.pop(22)
                data.append(labelDict[row[-1]])
                # data.insert(0, str(index))
                allData.append(data.copy())
                # index += 1

        index = 0
        poisonInex = random.sample([i for i in range(200 * 2, 250 * 2)], int(((250 - 200) * 0.2 * 2)))
        for item in allData:
            if str(int(item[-1])) == '2':
                continue
            if str(int(item[-1])) == '3':
                continue

            if selectCount[str(int(item[-1]))] < 75:
                data = item[:]
                data.insert(0, str(index))
                # if index in poisonInex:
                #     data[len(data) - 1] = ((int(data[len(data) - 1]) + 3) % 8)
                pretainData.append(data.copy())
                selectCount[str(int(item[-1]))] += 1
                index += 1
                continue
            elif selectCount[str(int(item[-1]))] < 200:
                data = item[:]
                data.insert(0, str(index))
                if index in poisonInex:
                    raC = random.random()
                    if raC > 0.5:
                        data[len(data) - 1] = ((int(data[len(data) - 1]) + 1) % 2)
                    else:
                        data[random.randint(1, 26)] = random.random()
                trainData.append(data.copy())
                selectCount[str(int(item[-1]))] += 1
                index += 1
                continue
            elif selectCount[str(int(item[-1]))] < 250:
                data = item[:]
                data.insert(0, str(index))
                # if index in poisonInex:
                #     data[random.randint(1, 26)] = random.random()
                #     data[random.randint(1, 26)] = random.random() * 10
                #     data[random.randint(1, 26)] = random.random() * 50
                #     data[random.randint(1, 26)] = random.random() * 100
                testData.append(data.copy())
                selectCount[str(int(item[-1]))] += 1
                index += 1
                continue

        pretainData = np.asarray(pretainData)
        tainData = np.asarray(trainData)
        testData = np.asarray(testData)

        # np.random.shuffle(trainData)

        np.save('./Tor-NonTor/pretrain.npy', pretainData)
        np.save('./Tor-NonTor/train.npy', tainData)
        np.save('./Tor-NonTor/test.npy', testData)

        return pretainData, tainData, testData

if __name__ == '__main__':
    # NovelBuildAllAndSplit()
    # NovelBuildRFAllAndSplit()
    # NovelBuildRFAllAndSplitShuffle()
    DarkNetCon()