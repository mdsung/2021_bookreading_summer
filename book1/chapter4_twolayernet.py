import numpy as np
from chapter3_activation import sigmoid, softmax
from chapter4_gradient import numerical_gradient
from chapter4_loss import cross_entropy_error

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        
        self.params["W1"] = weight_init_std * np.random.rand(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        
        self.params["W2"] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1 ,b2 = self.params["b1"], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)     
        
        return y 
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params["W1"])
        grads['b1'] = numerical_gradient(loss_W, self.params["b1"])
        grads['W2'] = numerical_gradient(loss_W, self.params["W2"])
        grads['b2'] = numerical_gradient(loss_W, self.params["b2"])
        
        return grads
    
if __name__ == '__main__':
    net = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)
    
    x = np.random.rand(100, 784)
    y = net.predict(x)
    t = np.random.rand(100, 10)
    
    grads = net.numerical_gradient(x, t)
    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)