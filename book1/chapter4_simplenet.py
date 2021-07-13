import os
import sys
import numpy as np
from chapter3_activation import softmax
from chapter4_loss import cross_entropy_error
from chapter4_gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    
    np.argmax(p)
    t = np.array([0, 0, 1])
    print(net.loss(x, t))
    
    dW = numerical_gradient(lambda w: net.loss(x, t), net.W)
    print(dW)

    