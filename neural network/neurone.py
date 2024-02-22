from random import random
import numpy as np
class neuroneRelu:
    def __init__(self, input_size, e):
        self.n1 = input_size
        self.w = np.full(input_size + 1, random())
        self.e = e
    def f(self, x):
        if x < 0:
            return 0
        else:
            return x
    def der_f(self, x):
        if x < 0:
            return 0
        else:
            return 1
    def forward(self, x):
        x = [1] + x
        x = np.ndarray(x)
        sum = np.dot(x.T, self.w[0])
        output = self.f(sum)
        output = np.dot(output, self.w[1])
        return sum, output
    def actualization(self, grad):
        self.w -= self.e * grad



class neuroneSigmoid:
    def __init__(self, input_size, output_size, e, f):
        self.n1 = input_size
        self.w = np.full(input_size + 1, random())
        self.e = e
    def f(self, x):
        return 1/(1+np.exp(-x))
    def der_f(self, x):
        return np.exp(-x) / ((1+np.exp(-x))**2)
    def forward(self, x):
        x = [1] + x
        x = np.ndarray(x)
        sum = np.dot(x.T, self.w[0])
        output = self.f(sum)
        output = np.dot(output, self.w[1])
        return sum, output
    def actualization(self, grad):
        self.w -= self.e * grad

class neuroneLinear:
    def __init__(self, input_size, e,):
        self.n1 = input_size
        self.w = np.full(input_size + 1, random())
        self.e = e
    def f(self, x):
        return x
    def der_f(self, x):
        return 1
    def forward(self, x):
        x = [1] + x
        x = np.ndarray(x)
        sum = np.dot(x.T, self.w[0])
        output = self.f(sum)
        output = np.dot(output, self.w[1])
        return sum, output
    def actualization(self, grad):
        self.w -= self.e * grad
