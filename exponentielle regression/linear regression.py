from neurone import neurone
from cost import cost
import matplotlib.pyplot as plt
import random
import numpy as np

nbr = 1100

X = [i for i in range(nbr)]
Y = [np.exp(-i + random.uniform(-1, 1)) for i in range(nbr)]

plt.figure()

plt.show()

pas = 0.001
nbr_epochs = 10000
nbr_input = 1
couts = []
net = neurone(nbr_input, pas)
for _ in range(nbr_epochs):
    for i in range(nbr):
        #on prédit le résultat avec le réseau de neurone #1
        y_pred = net.forward([X[i]])
        y = Y[i]
        #on calcul le coup #2
        cout = cost([X[i]], [y_pred], [y])
        valeur = cout.value()
        couts.append(valeur)
        #on calcul le grad du couts #3
        grad = cout.gradient()

        print(net.w)
        #on actualise le réseau de neurones #3
        net.actualization(grad)

Y_pred = []
for i in range(nbr):
    Y_pred.append(net.forward([X[i]]))

plt.figure()
plt.plot(X, Y_pred, color ='b', label = f"régression linéaire a = {round(net.w[1], 3)}, b = {round(net.w[0], 3)} tel que y = {round(net.w[1], 3)}x + {round(net.w[0], 3)}")
plt.scatter(X, Y, color ='y')
plt.legend()
plt.show()
plt.figure()
plt.plot([i for i in range(len(couts))], couts, color ='r')
plt.show()
