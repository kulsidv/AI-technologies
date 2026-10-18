import torch
import torch.nn as nn
from tensorflow.keras.datasets import mnist


class NET(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(75, 10),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterian = nn.MSELoss()
        self.epoches = 10

    def one_hot_encoding(self, y):
        result = torch.zeros((1, 10))
        result[0][y] = 1
        return result

    def fit(self, X, y):
        self.model.train()
        X = torch.tensor(X / 255.0, dtype=torch.float, device=self.device).reshape(-1, 1, 1, 28, 28)
        for epoche in range(self.epoches):
            accuracy = 0
            for i, x in enumerate(X):
                output = self.model(x)
                loss = self.criterian(output, self.one_hot_encoding(y[i]))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accuracy += 1 if torch.argmax(output) == y[i] else 0
            print(accuracy/len(X))


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    net = NET()
    net.fit(X_train, y_train)


if __name__ == "__main__":
    main()
