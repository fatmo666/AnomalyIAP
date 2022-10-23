

# ------------------ LSTM-DC ----------------- #
class LSTM_DC(nn.Module):
    def __init__(self, noClass, eps, min_samples):
        super(LSTM_DC, self).__init__()

        # DBSCAN 定义
        self.eps = eps
        self.min_samples = min_samples
        self.cluster = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def forward(self, x):

        clustring = self.cluster.fit(multiOut)

        return out