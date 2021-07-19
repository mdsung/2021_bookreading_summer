import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=2021)

def relu(x):
    return np.maximum(0, x)

x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    
    w = np.random.rand(node_num, node_num) * (np.sqrt(2/node_num))
    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

figure = plt.gcf()
figure.set_size_inches(16,8)
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(f'{i+1}-layers')
    plt.hist(a.flatten(), 30, range = (0, 1))

plt.savefig(f'book1/figure/chapter6_initalize_he_relu.png', dpi = 300)