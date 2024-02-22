from neurone import neuroneRelu
from neurone import neuroneLinear
from neurone import neuroneSigmoid
import numpy as np
class layerReLu():
    def __init__(self, input_size, output_size, e):
        self.input_size = input_size
        self.output_size = output_size
        self.e = e
        self.layer = np.array([neuroneRelu(input_size, e) for _ in range(output_size)])
    def forward(self, x):
        return [self.layer[i].forward(x) for i in range(self.output_size)]
    def actualization(self, grads):
        for i in range(self.output_size):
            self.layer[i].actualization(grads[i])

class layerSigmoid():
    def __init__(self, input_size, output_size, e):
        self.input_size = input_size
        self.output_size = output_size
        self.e = e
        self.layer = np.array([neuroneSigmoid(input_size, e) for _ in range(output_size)])
    def forward(self, x):
        return [self.layer[i].forward(x) for i in range(self.output_size)]
    def actualization(self, grads):
        for i in range(self.output_size):
            self.layer[i].actualization(grads[i])

class layerLinear():
    def __init__(self, input_size, output_size, e):
        self.input_size = input_size
        self.output_size = output_size
        self.e = e
        self.layer = np.array([neuroneLinear(input_size, e) for _ in range(output_size)])
    def forward(self, x):
        return [self.layer[i].forward(x) for i in range(self.output_size)]
    def actualization(self, grads):
        for i in range(self.output_size):
            self.layer[i].actualization(grads[i])
    def backward(self, grads):
        newgrad = []
        for i in range(self.output_size):
            newgrad.append([grads[i][j] * self.layer[i].der_f(grads[i][j]) for j in range(len(grads[i]))])
        return newgrad