import numpy as np

def function_2(x):
    return x[0] ** 2 + x[1] ** 2 

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    for element in it:
        idx = it.multi_index
        tmp_val = element
        
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 값 복원
        
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    for _ in range(step_num):
        # grad = np.gradient(f, x)
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([2., 3.])))
