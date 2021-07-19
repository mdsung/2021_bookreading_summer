import numpy as np

class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.droupout_ratio
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio) # 곱하지 않아도 된다. 
        
    def backward(self, dout):
        return dout * self.mask
