import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=2021)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t

def relu(x):
    return np.maximum(0, x)

activation_function = "sigmoid"
activation_functions = {"sigmoid": sigmoid, 
                        "tanh":tanh,
                        "relu":relu}

x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    
    w = np.random.rand(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x, w)
    
    z = activation_functions[activation_function](a)
    activations[i] = z

figure = plt.gcf()
figure.set_size_inches(16,8)
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(f'{i+1}-layers')
    plt.hist(a.flatten(), 30, range = (0, 1))

plt.savefig(f'book1/figure/chapter6_initalize_xavier_{activation_function}.png', dpi = 300)