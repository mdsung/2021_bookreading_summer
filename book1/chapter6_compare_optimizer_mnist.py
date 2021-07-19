import numpy as np
import matplotlib.pyplot as plt

from dataset import load_mnist
from chapter6_optimizer import * 
from chapter5_twolayernet import TwoLayerNet
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 100
max_iteration = 2000

optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
    train_loss[key] = []
    
for i in range(max_iteration):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)

    if i % 100 == 0:
        print(f"===========iteration:{i}===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(f"{key}:{loss}")
            
markers = {"SGD":"o", "Momentum":'x', "AdaGrad":"s", "Adam":"D"}
x = np.arange(max_iteration)
for key in optimizers.keys():
    plt.plot(x, train_loss[key], marker = markers[key], markevery = 100, label = key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0,1)
plt.legend()
plt.savefig('optimizer_compare_mnist.png', dpi=300)
