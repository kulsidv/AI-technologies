"""
Функция активации - ReLU
На выходе - Softmax
Функция потерь - Cross-Enthropy
+ График функции потерь и точности
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

EPOCHE_COUNT = 10


def one_hot_encodding(y):
    one_hot = np.zeros(10)
    one_hot[y] = 1
    return one_hot.reshape(1, -1)


class Net:
    def __init__(self):
        self.weight_input_hidden = np.random.random((28 * 28, 128))
        self.weight_output_hidden = np.random.random((128, 10))
        self.lr = 0.01
        self.hidden = np.zeros((1, 128))

    def forward_propagation(self, X):
        X = X.reshape(1, -1)
        self.hidden = self.reLU(X @ self.weight_input_hidden)  # (1*128)
        return self.softmax(self.hidden @ self.weight_output_hidden)  # (1*10)

    def reLU(self, layer):
        return np.maximum(0, layer)

    def softmax(self, layer):
        exp_layer = np.exp(layer - np.max(layer))
        return exp_layer / np.sum(exp_layer)

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output)

    def back_propagation(self, input, output, y):
        one_hot = one_hot_encodding(y)
        self.weight_input_hidden -= (
            self.lr
            * input.T
            @ (output - one_hot)
            @ self.weight_output_hidden.T
            * (self.hidden > 0)
        )
        self.weight_output_hidden -= self.lr * self.hidden.T @ (output - one_hot)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    net = Net()
    losses = []
    accuracy = []
    for epoche in range(EPOCHE_COUNT):
        epoche_losses = []
        epoche_accuracy = 0
        for x, y in zip(X_train, y_train):
            output = net.forward_propagation(x)
            epoche_losses.append(-np.sum(one_hot_encodding(y) * np.log(output + 1e-10)))
            epoche_accuracy += 1 if np.argmax(output) == y else 0
            net.back_propagation(x.reshape(1, -1), output, y)
        losses.append(sum(epoche_losses)/len(X_train))
        accuracy.append(epoche_accuracy/len(X_train))

    plt.plot(range(1, EPOCHE_COUNT+1), losses, label='Функция потерь')
    plt.plot(range(1, EPOCHE_COUNT+1), accuracy, label='Точность')
    plt.xlabel('Эпохи')
    plt.legend()
    plt.show()
