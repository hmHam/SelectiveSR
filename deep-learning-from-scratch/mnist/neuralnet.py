import pickle
import numpy as np
from functions import sigmoid, softmax
from mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist()
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    z1 = sigmoid(np.dot(x, W1) + b1)
    z2 = sigmoid(np.dot(z1, W2) + b2)
    y = softmax(np.dot(z2, W3) + b3)
    return y


if __name__ == '__main__':
    x, t = get_data()
    network = init_network()
    BATCH_SIZE = 100
    accuracy_count = 0
    for i in range(0, len(x), BATCH_SIZE):
        y = predict(network, x[i:i+BATCH_SIZE])
        p = np.argmax(y)
        accuracy_count += np.sum(p == t[i:i+BATCH_SIZE])
    print('Accuracy:', float(accuracy_count) / len(x))
