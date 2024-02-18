from neurone import neurone
class layer():
    def __init__(self, input_size, output_size, e, f):
        self.output_size = output_size
        self.layer = [neurone(input_size, e,f) for _ in range(output_size)]
    def forward(self, x):
        output = []
        for net in self.layer:
            output.append(net.forward(x))
        return output
    def actualization(self, grads):
        for net, grad in zip(self.layer, grads):
            net.actualization(grad)