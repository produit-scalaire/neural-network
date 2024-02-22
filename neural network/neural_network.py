from layer import layerReLu
from  layer import layerLinear
from layer import layerSigmoid
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, e):
        self.layer1 = layerLinear(input_size, hidden_size, e)
        self.layer2 = layerReLu(hidden_size, hidden_size, e)
        self.layer3 = layerLinear(hidden_size, output_size, e)
        self.network = [self.layer1, self.layer2, self.layer3]
    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        return x
    def actualization(self, grads):
        for layer in self.network:
            layer.actualization(grads)
            grads = layer.backward(grads)
