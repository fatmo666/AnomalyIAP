import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as Data
import more_itertools
from utils import validate_by_rmse, Normalizer, compute_precision_and_recall
from sklearn.model_selection import train_test_split

# feature_size = 26
feature_size = 79
seq_len = 1
batch_size = 2048
# epoches = 50

lr = 1e-3
weight_decay = 1e-5

class LstmModel():
    def __init__(self, classNum=2):
        self.model = LSTM_multivariate(classNum)
        self.optimizier = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def trainModel(self, sample, label, epoches=50):
        """
        训练模型
        """

        indexs = list(sample[:, 0])
        trainData = sample[:, 1:]
        # trainData = sample

        # ------------------Rnn预训练----------------- #
        criterion = nn.MSELoss()
        getMSEvec = nn.MSELoss(reduction='none')
        loss_func = nn.CrossEntropyLoss()
        # normer = Normalizer(trainData.shape[-1], online_minmax=False)
        # train_feat = normer.fit_transform(trainData)
        self.model = trainRnn(model=self.model, optimizier=self.optimizier, loss_func=loss_func, train_data=trainData, label=label, epoches=epoches)

        # state = {'net': self.model.state_dict(), 'optimizer': self.optimizier.state_dict(), 'epoch': epoches}
        # torch.save(state, './model/lstm.pth')


    def inputData(self, preTrainPath='./Tor-NonTor/pretrain.npy', trainPath='./Tor-NonTor/train.npy', testPath='./Tor-NonTor/test.npy'):
        """
        输入数据
        """
        preTrainDataLoad = np.load(preTrainPath)
        trainDataLoad = np.load(trainPath)
        testrainDataLoad = np.load(testPath)

        preTrainData = np.squeeze(np.asarray(preTrainDataLoad[:, :-1], dtype=float))
        preTrainLabel = np.squeeze(np.asarray(preTrainDataLoad[:, -1:], dtype=int))

        trainData = np.squeeze(np.asarray(trainDataLoad[:, :-1], dtype=float))
        trainLabel = np.squeeze(np.asarray(trainDataLoad[:, -1:], dtype=int))

        testData = np.squeeze(np.asarray(testrainDataLoad[:, :-1], dtype=float))
        testLabel = np.squeeze(np.asarray(testrainDataLoad[:, -1:], dtype=int))


        return preTrainData, preTrainLabel, trainData, trainLabel, testData, testLabel

    # def inputData(self, trainPath='./Tor-novel/train.npy', testPath='./Tor-novel/test.npy', allPath='./Tor-novel/all.npy'):
    #     """
    #     输入数据
    #     """
    #     labeledDataLoad = np.load(trainPath)
    #     unlabeledDataLoad = np.load(testPath)
    #     allDataLoad = np.load(allPath)
    #
    #     labeledData = np.squeeze(np.asarray(labeledDataLoad[:, :-1], dtype=float))
    #     unlabeledData = np.squeeze(np.asarray(unlabeledDataLoad[:, :-1], dtype=float))
    #
    #     labeledLabel = np.squeeze(np.asarray(labeledDataLoad[:, -1:], dtype=int))
    #     unlabeledLabel = np.squeeze(np.asarray(unlabeledDataLoad[:, -1:], dtype=int))
    #
    #     trainData = np.squeeze(np.asarray(allDataLoad[:, :-1], dtype=float))
    #     trainLabel = np.squeeze(np.asarray(allDataLoad[:, -1:], dtype=int))
    #
    #     return labeledData, unlabeledData, labeledLabel, unlabeledLabel, labeledDataLoad, unlabeledDataLoad

    def predict(self, sample):
        """
        预测样本
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        indexs = sample[:, 0]
        sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, np.argmax(rnn_sample, axis=1)

    def probability(self, sample):
        """
        获取样本概率值
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()
        probs = torch.nn.functional.softmax(preOut, dim=1)

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return probs.detach().cpu().numpy()

    def probabilityForAopc(self, sample):
        """
        获取样本概率值
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        # sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()
        probs = torch.nn.functional.softmax(preOut, dim=1)

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return probs.detach().cpu().numpy()

    def featureExtract(self, sample):
        """
        特征提取
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        indexs = sample[:, 0]
        sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, cluster_sample

    def featureExtractOne(self, sample):
        """
        特征提取
        """

        indexs = []
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        # X_train = X_train.view(len(X_train), 1, -1)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, cluster_sample

    def loadModel(self, modelPath='./model/lstm.pth'):
        """
        读取模型
        """
        checkpoint = torch.load(modelPath)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizier.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

class LstmModel02():
    def __init__(self, classNum=8):
        self.model = LSTM_multivariate02(classNum)
        self.optimizier = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def trainModel(self, sample, label, epoches=50):
        """
        训练模型
        """

        indexs = list(sample[:, 0])
        trainData = sample[:, 1:]
        # trainData = sample

        # ------------------Rnn预训练----------------- #
        criterion = nn.MSELoss()
        getMSEvec = nn.MSELoss(reduction='none')
        loss_func = nn.CrossEntropyLoss()
        # normer = Normalizer(trainData.shape[-1], online_minmax=False)
        # train_feat = normer.fit_transform(trainData)
        self.model = trainRnn(model=self.model, optimizier=self.optimizier, loss_func=loss_func, train_data=trainData, label=label, epoches=epoches)

        # state = {'net': self.model.state_dict(), 'optimizer': self.optimizier.state_dict(), 'epoch': epoches}
        # torch.save(state, './model/lstm.pth')


    def inputData(self, preTrainPath='./Tor-novel/pretrain.npy', trainPath='./Tor-novel/train.npy', testPath='./Tor-novel/test.npy'):
        """
        输入数据
        """
        preTrainDataLoad = np.load(preTrainPath)
        trainDataLoad = np.load(trainPath)
        testrainDataLoad = np.load(testPath)

        preTrainData = np.squeeze(np.asarray(preTrainDataLoad[:, :-1], dtype=float))
        preTrainLabel = np.squeeze(np.asarray(preTrainDataLoad[:, -1:], dtype=int))

        trainData = np.squeeze(np.asarray(trainDataLoad[:, :-1], dtype=float))
        trainLabel = np.squeeze(np.asarray(trainDataLoad[:, -1:], dtype=int))

        testData = np.squeeze(np.asarray(testrainDataLoad[:, :-1], dtype=float))
        testLabel = np.squeeze(np.asarray(testrainDataLoad[:, -1:], dtype=int))


        return preTrainData, preTrainLabel, trainData, trainLabel, testData, testLabel

    # def inputData(self, trainPath='./Tor-novel/train.npy', testPath='./Tor-novel/test.npy', allPath='./Tor-novel/all.npy'):
    #     """
    #     输入数据
    #     """
    #     labeledDataLoad = np.load(trainPath)
    #     unlabeledDataLoad = np.load(testPath)
    #     allDataLoad = np.load(allPath)
    #
    #     labeledData = np.squeeze(np.asarray(labeledDataLoad[:, :-1], dtype=float))
    #     unlabeledData = np.squeeze(np.asarray(unlabeledDataLoad[:, :-1], dtype=float))
    #
    #     labeledLabel = np.squeeze(np.asarray(labeledDataLoad[:, -1:], dtype=int))
    #     unlabeledLabel = np.squeeze(np.asarray(unlabeledDataLoad[:, -1:], dtype=int))
    #
    #     trainData = np.squeeze(np.asarray(allDataLoad[:, :-1], dtype=float))
    #     trainLabel = np.squeeze(np.asarray(allDataLoad[:, -1:], dtype=int))
    #
    #     return labeledData, unlabeledData, labeledLabel, unlabeledLabel, labeledDataLoad, unlabeledDataLoad

    def predict(self, sample):
        """
        预测样本
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        indexs = sample[:, 0]
        sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, np.argmax(rnn_sample, axis=1)

    def probability(self, sample):
        """
        获取样本概率值
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()
        probs = torch.nn.functional.softmax(preOut, dim=1)

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return probs.detach().cpu().numpy()

    def featureExtract(self, sample):
        """
        特征提取
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        indexs = sample[:, 0]
        sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, cluster_sample

    def featureExtractOne(self, sample):
        """
        特征提取
        """

        indexs = []
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        # X_train = X_train.view(len(X_train), 1, -1)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, cluster_sample

    def loadModel(self, modelPath='./model/lstm02.pth'):
        """
        读取模型
        """
        checkpoint = torch.load(modelPath)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizier.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

class LstmModel03():
    def __init__(self, classNum=8):
        self.model = LSTM_multivariate03(classNum)
        self.optimizier = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def trainModel(self, sample, label, epoches=50):
        """
        训练模型
        """

        indexs = list(sample[:, 0])
        trainData = sample[:, 1:]
        # trainData = sample

        # ------------------Rnn预训练----------------- #
        criterion = nn.MSELoss()
        getMSEvec = nn.MSELoss(reduction='none')
        loss_func = nn.CrossEntropyLoss()
        # normer = Normalizer(trainData.shape[-1], online_minmax=False)
        # train_feat = normer.fit_transform(trainData)
        self.model = trainRnn(model=self.model, optimizier=self.optimizier, loss_func=loss_func, train_data=trainData, label=label, epoches=epoches)

        # state = {'net': self.model.state_dict(), 'optimizer': self.optimizier.state_dict(), 'epoch': epoches}
        # torch.save(state, './model/lstm.pth')


    def inputData(self, preTrainPath='./Tor-novel/pretrain.npy', trainPath='./Tor-novel/train.npy', testPath='./Tor-novel/test.npy'):
        """
        输入数据
        """
        preTrainDataLoad = np.load(preTrainPath)
        trainDataLoad = np.load(trainPath)
        testrainDataLoad = np.load(testPath)

        preTrainData = np.squeeze(np.asarray(preTrainDataLoad[:, :-1], dtype=float))
        preTrainLabel = np.squeeze(np.asarray(preTrainDataLoad[:, -1:], dtype=int))

        trainData = np.squeeze(np.asarray(trainDataLoad[:, :-1], dtype=float))
        trainLabel = np.squeeze(np.asarray(trainDataLoad[:, -1:], dtype=int))

        testData = np.squeeze(np.asarray(testrainDataLoad[:, :-1], dtype=float))
        testLabel = np.squeeze(np.asarray(testrainDataLoad[:, -1:], dtype=int))


        return preTrainData, preTrainLabel, trainData, trainLabel, testData, testLabel

    # def inputData(self, trainPath='./Tor-novel/train.npy', testPath='./Tor-novel/test.npy', allPath='./Tor-novel/all.npy'):
    #     """
    #     输入数据
    #     """
    #     labeledDataLoad = np.load(trainPath)
    #     unlabeledDataLoad = np.load(testPath)
    #     allDataLoad = np.load(allPath)
    #
    #     labeledData = np.squeeze(np.asarray(labeledDataLoad[:, :-1], dtype=float))
    #     unlabeledData = np.squeeze(np.asarray(unlabeledDataLoad[:, :-1], dtype=float))
    #
    #     labeledLabel = np.squeeze(np.asarray(labeledDataLoad[:, -1:], dtype=int))
    #     unlabeledLabel = np.squeeze(np.asarray(unlabeledDataLoad[:, -1:], dtype=int))
    #
    #     trainData = np.squeeze(np.asarray(allDataLoad[:, :-1], dtype=float))
    #     trainLabel = np.squeeze(np.asarray(allDataLoad[:, -1:], dtype=int))
    #
    #     return labeledData, unlabeledData, labeledLabel, unlabeledLabel, labeledDataLoad, unlabeledDataLoad

    def predict(self, sample):
        """
        预测样本
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        indexs = sample[:, 0]
        sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, np.argmax(rnn_sample, axis=1)

    def probability(self, sample):
        """
        获取样本概率值
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()
        probs = torch.nn.functional.softmax(preOut, dim=1)

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return probs.detach().cpu().numpy()

    def featureExtract(self, sample):
        """
        特征提取
        """
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        indexs = sample[:, 0]
        sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, cluster_sample

    def featureExtractOne(self, sample):
        """
        特征提取
        """

        indexs = []
        # 使用rnn提取特征
        # X_train = normer.fit_transform(sample)
        # X_train = more_itertools.windowed(X_train, n=seq_len, step=1)
        sample = sample.numpy()
        # indexs = sample[:, 0]
        # sample = sample[:, 1:]
        X_train = more_itertools.windowed(sample, n=seq_len, step=1)
        X_train = np.asarray(list(X_train))
        X_train = torch.from_numpy(X_train).type(torch.float)

        # X_train = X_train.view(len(X_train), 1, -1)

        out, preOut, multiOut = self.model(X_train)
        rnn_sample = preOut.cpu().data.numpy()
        cluster_sample = multiOut.cpu().data.numpy()

        # print(np.argmax(rnn_sample, axis=1))

        # print("lstm acc: ", accuracy_score(np.argmax(rnn_sample, axis=1), labelStruck))

        return indexs, cluster_sample

    def loadModel(self, modelPath='./model/lstm03.pth'):
        """
        读取模型
        """
        checkpoint = torch.load(modelPath)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizier.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

# ------------------ LSTM ----------------- #
class LSTM_multivariate(nn.Module):
    def __init__(self, noClass):
        super(LSTM_multivariate, self).__init__()

        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=16,  # rnn hidden unit
            # hidden_size=64,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # dropout=0.3
        )

        # 排列标签
        self.out = nn.Linear(16, noClass)
        # self.out = nn.Linear(64, noClass)
        # self.out = nn.Linear(64, feature_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        # preOut = torch.cat((r_out[1,:-1,:], r_out[:,-1,:]),0)
        multiOut = torch.cat((r_out[0, :-1, :], r_out[:, -1, :]), 0)
        preOut = torch.cat((r_out[0, :-1, :], r_out[:, -1, :]), 0)
        out = self.out(r_out[:, -1, :])
        preOut = self.out(preOut)
        return out, preOut, multiOut

def trainRnn(model, optimizier, loss_func, train_data, label, epoches=30):
    model.train()

    X_train = more_itertools.windowed(train_data, n=seq_len, step=1)
    X_train = np.asarray(list(X_train))

    y_train = more_itertools.windowed(label, n=seq_len, step=1)
    y_train = np.asarray(list(y_train))

    # X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    # y_train = torch.from_numpy(y_train).type(torch.int64).to(device)
    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.int64)

    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output, preOut, multiOut = model(batch_x)  # rnn output
            loss = loss_func(output, batch_y[:, -1])  # cross entropy loss
            optimizier.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizier.step()  # apply gradients

            if step % 1000 == 0:
                test_output, preOut, multiOut = model(batch_x)  # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                accuracy = float((pred_y == np.asarray(batch_y[:, -1].cpu())).astype(int).sum()) / float(np.asarray(batch_y[:, -1].cpu()).size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    return model

# ------------------ LSTM ----------------- #
class LSTM_multivariate02(nn.Module):
    def __init__(self, noClass):
        super(LSTM_multivariate02, self).__init__()

        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=32,  # rnn hidden unit
            # hidden_size=64,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # dropout=0.3
        )

        # 排列标签
        self.out = nn.Linear(32, noClass)
        # self.out = nn.Linear(64, noClass)
        # self.out = nn.Linear(64, feature_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        # preOut = torch.cat((r_out[1,:-1,:], r_out[:,-1,:]),0)
        multiOut = torch.cat((r_out[0, :-1, :], r_out[:, -1, :]), 0)
        preOut = torch.cat((r_out[0, :-1, :], r_out[:, -1, :]), 0)
        out = self.out(r_out[:, -1, :])
        preOut = self.out(preOut)
        return out, preOut, multiOut

def trainRnn(model, optimizier, loss_func, train_data, label, epoches=30):
    model.train()

    X_train = more_itertools.windowed(train_data, n=seq_len, step=1)
    X_train = np.asarray(list(X_train))

    y_train = more_itertools.windowed(label, n=seq_len, step=1)
    y_train = np.asarray(list(y_train))

    # X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    # y_train = torch.from_numpy(y_train).type(torch.int64).to(device)
    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.int64)

    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output, preOut, multiOut = model(batch_x)  # rnn output
            loss = loss_func(output, batch_y[:, -1])  # cross entropy loss
            optimizier.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizier.step()  # apply gradients

            if step % 1000 == 0:
                test_output, preOut, multiOut = model(batch_x)  # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                accuracy = float((pred_y == np.asarray(batch_y[:, -1].cpu())).astype(int).sum()) / float(np.asarray(batch_y[:, -1].cpu()).size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    return model

# ------------------ LSTM ----------------- #
class LSTM_multivariate03(nn.Module):
    def __init__(self, noClass):
        super(LSTM_multivariate03, self).__init__()

        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=8,  # rnn hidden unit
            # hidden_size=64,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # dropout=0.3
        )

        # 排列标签
        self.out = nn.Linear(8, noClass)
        # self.out = nn.Linear(64, noClass)
        # self.out = nn.Linear(64, feature_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        # preOut = torch.cat((r_out[1,:-1,:], r_out[:,-1,:]),0)
        multiOut = torch.cat((r_out[0, :-1, :], r_out[:, -1, :]), 0)
        preOut = torch.cat((r_out[0, :-1, :], r_out[:, -1, :]), 0)
        out = self.out(r_out[:, -1, :])
        preOut = self.out(preOut)
        return out, preOut, multiOut

def trainRnn(model, optimizier, loss_func, train_data, label, epoches=30):
    model.train()

    X_train = more_itertools.windowed(train_data, n=seq_len, step=1)
    X_train = np.asarray(list(X_train))

    y_train = more_itertools.windowed(label, n=seq_len, step=1)
    y_train = np.asarray(list(y_train))

    # X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    # y_train = torch.from_numpy(y_train).type(torch.int64).to(device)
    X_train = torch.from_numpy(X_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.int64)

    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output, preOut, multiOut = model(batch_x)  # rnn output
            loss = loss_func(output, batch_y[:, -1])  # cross entropy loss
            optimizier.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizier.step()  # apply gradients

            if step % 1000 == 0:
                test_output, preOut, multiOut = model(batch_x)  # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                accuracy = float((pred_y == np.asarray(batch_y[:, -1].cpu())).astype(int).sum()) / float(np.asarray(batch_y[:, -1].cpu()).size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    return model

if __name__ == '__main__':
    lstmModel = LstmModel()

    # lstmModel.trainModel()
    lstmModel.loadModel()

    preTrainData, preTrainLabel, trainData, trainLabel, testData, testLabel = lstmModel.inputData()
    # lstmModel.featureExtractOne(labeledData[0])
    indexs, result = lstmModel.predict(testData)
    print(indexs, result)