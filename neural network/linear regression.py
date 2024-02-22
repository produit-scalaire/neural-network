import torch
import torchvision
import torchvision.transforms as transforms
from neural_network import NeuralNetwork

# Charger les données MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialiser le réseau de neurones
input_size = 784  # Taille des images MNIST (28x28)
hidden_size = 128  # Taille de la couche cachée
output_size = 10  # Nombre de classes (0-9)
e = 0.01  # Taux d'apprentissage
network = NeuralNetwork(input_size, hidden_size, output_size, e)

# Boucle d'entraînement
for epoch in range(2):  # Boucle sur l'ensemble de données plusieurs fois
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Obtenir les entrées ; data est une liste de [inputs, labels]
        inputs, labels = data
        inputs = inputs.view(inputs.shape[0], -1)  # Redimensionner les images en vecteurs

        # Propagation avant
        outputs = network.forward(inputs)
        # Calculer la perte
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        # Propagation arrière et optimisation
        loss.backward()
        network.actualization(loss)

        # Imprimer les statistiques
        running_loss += loss.item()
        if i % 2000 == 1999:  # Imprimer toutes les 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Entraînement terminé')
