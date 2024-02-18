import numpy as np

class Neuron:
    def __init__(self, input_size, output_size):
        # Initialize weights and bias
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)

    def forward(self, inputs):
        # Compute weighted sum
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function (e.g., sigmoid) to each element of the array
        output = self.sigmoid(weighted_sum)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Example usage
input_size = 3
output_size = 15
neuron = Neuron(input_size, output_size)
inputs = np.array([0.5, 0.3, 0.2])
output = neuron.forward(inputs)
print("Output:", output)