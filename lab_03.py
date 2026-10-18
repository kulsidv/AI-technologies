import torch
import torch.nn as nn
from tensorflow.keras.datasets import mnist


class NET(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterian = nn.MSELoss()
        self.epoches = 10

    def one_hot_encoding(self, y):
        result = torch.zeros((1, 10))
        result[0][y] = 1
        return result

    def fit(self, X, y):
        self.model.train()
        X = torch.tensor(X, dtype=torch.float, device=self.device).reshape(-1, 1, 28*28)
        for epoche in range(self.epoches):
            loss_sum = 0
            for i, x in enumerate(X):
                output = self.model(x)
                loss = self.criterian(output, self.one_hot_encoding(y[i]))
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            print(loss/len(X))


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    net = NET()
    net.fit(X_train[:100], y_train[:100])


if __name__ == "__main__":
    main()
