#%%
import numpy as np
from chapter4_twolayernet import TwoLayerNet
from dataset import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size = 784, hidden_size = 500, output_size = 10)

train_lost_list = []
train_acc_list = []
test_acc_list = []

ITER_NUM = 1000
TRAIN_SIZE = x_train.shape[0]
BATCH_SIZE = 100
LEARNING_RATE = 0.1

iter_per_epoch = max(TRAIN_SIZE / BATCH_SIZE, 1)

for i in range(ITER_NUM):
    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= LEARNING_RATE * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_lost_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_test)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")
