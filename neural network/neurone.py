from random import random
import numpy as np
class neurone:
    def __init__(self, input_size, output_size, e, f):
        self.n1 = input_size
        self.w = np.full(input_size + 1, random())
        self.e = e
        self.f = f
    def forward(self, x):
        x = [1] + x
        x = np.ndarray(x)
        sum = np.dot(x.T, self.w)
        output = self.f(sum)
        return sum, output
    def actualization(self, grad):
        self.w -= self.e * grad