import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class Net:
    def __init__(self):
        self.weight_input_hidden = np.random.random((28 * 28, 128))
        self.weight_output_hidden = np.random.random((128, 10))
        self.lr = 0.01
        self.hidden = np.array((1, 128))
        self.output = np.array((1, 10))

    def forward_propagation(self, X):
        X = X.reshape(1, -1)
        self.hidden = self.sigmoid(X @ self.weight_input_hidden)  # (1*128)
        self.output = self.sigmoid(self.hidden @ self.weight_output_hidden)  # (1*10)

    def sigmoid(self, layer):
        return 1 / (1 + np.exp(-layer))

    def predict(self, X):
        self.output = self.forward_propagation(X)
        return np.argmax(self.output)

    # для обратного распространения - отметить идеальный вариант и вычитать потом
    def one_hot_encodding(self, y):
        one_hot = np.zeros(10)
        one_hot[y] = 1
        return one_hot.reshape(1, -1)

    def back_propagation(self, input, y):
        one_hot = self.one_hot_encodding(y)
        self.weight_input_hidden -= (
            2
            * self.lr
            * input.T
            @ (self.output * (self.output - one_hot) * (1 - self.output))
            @ self.weight_output_hidden
            * (self.hidden * (1 - self.hidden))
        )
        # матричная запись производной
        self.weight_output_hidden -= (
            2
            * self.lr
            * self.hidden.T
            @ (self.output * (self.output - one_hot))
            * (1 - self.output)

        )


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    net = Net()
    net.back_propagation(X_train[0].reshape(1, -1), y_train[0])