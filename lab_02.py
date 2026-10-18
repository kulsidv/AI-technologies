import torch
import torch.nn as nn
from tensorflow.keras.datasets import mnist


class NET:
    def __init__(self):
        self.weight = [
            torch.rand((28 * 28, 128), dtype=torch.float, requires_grad=True),
            torch.rand((128, 10), dtype=torch.float, requires_grad=True),
        ]
        self.lr = 10
        self.epoches = 100

    def activation(self, layer):
        return torch.sigmoid(layer)

    def one_hot_encoding(self, y):
        result = torch.zeros((1, 10))
        result[0][y] = 1
        return result

    def mse_loss(self, output, y):
        return torch.sum((output - self.one_hot_encoding(y)) ** 2)

    def forward_propagation(self, input):
        output = input
        for i in self.weight:
            output = output @ i
            output = self.activation(output)
        return output

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float).reshape(-1, 1, 28 * 28)
        for epoche in range(self.epoches):
            loses = 0
            for i, x in enumerate(X):
                output = self.forward_propagation(x)
                mse_loss = self.mse_loss(output, y[i])
                loses += mse_loss
                self.back_propagation(mse_loss)
            print(loses / len(X))

    def back_propagation(self, mse_loss):
        mse_loss.backward()
        with torch.no_grad():
            for w in self.weight:
                w -= self.lr * w.grad
                w.grad.zero_()

    def predict(self, X):
        return [torch.argmax(self.forward_propagation(x)) for x in X]


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    net = NET()
    net.fit(X_train, y_train)


if __name__ == "__main__":
    main()
