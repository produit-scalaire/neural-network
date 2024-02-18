from random import random
class neurone:
    def __init__(self, nbr_input, e):
        self.n = nbr_input
        self.w = [random() for _ in range(nbr_input + 1)]
        self.e = e
    def forward(self, x):
        x = [1] + x
        sum = 0
        for i in range(self.n + 1):
            sum += x[i] * self.w[i]
        return sum
    def actualization(self, grad):
        for i in range(self.n + 1):
            self.w[i] -= self.e * grad[i]