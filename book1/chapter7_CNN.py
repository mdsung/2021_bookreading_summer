import time
import numpy as np
from collections import OrderedDict

from chapter7_convolutional_layer import Convolution
from chapter7_pooling_layer import Pooling
from chapter5_layers import Affine, Relu, SoftmaxWithLoss
from dataset import load_mnist

class SimpleConvNet:
    def __init__(self, 
                input_dim = (1, 28, 28),
                conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size = 100,
                output_size = 10,
                weight_init_std = 0.01
                ):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']

        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.rand(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h = 2, pool_w = 2, stride = 2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x =  layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        
        grads ={}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

if __name__=='__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True)
    
    tic = time.process_time()
    network = SimpleConvNet()

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    ITER_NUM = 10000
    TRAIN_SIZE = x_train.shape[0]
    BATCH_SIZE = 100
    LEARNING_RATE = 0.1

    iter_per_epoch = max(TRAIN_SIZE / BATCH_SIZE, 1)

    for i in range(ITER_NUM):
        batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            network.params[key] -= LEARNING_RATE * grad[key]
            
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_test)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train acc, test acc | {train_acc}, {test_acc}")

    toc = time.process_time()
    print(toc - tic)

        