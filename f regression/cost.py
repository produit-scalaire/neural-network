class cost:
    def __init__(self, x, y_pred, y, f, der_f):
        self.x = [1] + x
        self.y = y
        self.y_pred = y_pred
        self.n1 = len(self.x)
        self.n2 = len(self.y)
        self.f = f
        self.der_f = der_f
    def value(self):
        sum = 0
        for i in range(self.n2):
            sum += (self.y_pred[i][1] - self.y[i]) ** 2
        return sum
    def gradient(self):
        grad = []
        for j in range(self.n1):
            der = 0
            for i in range(self.n2):
                der += 2 * self.x[j] * self.der_f(self.y_pred[i][0]) * (self.y_pred[i][1] - self.y[i])
            grad.append(der/self.n2)
        return grad