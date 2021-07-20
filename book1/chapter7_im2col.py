#%%
import numpy as np

def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    ## 1, 2-dimension에는 padding 하지 않고, 
    ## image layer인 3, 4-dimension에만 padding
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose[0, 4, 5, 1, 2 ,3].reshape[N * out_h * out_w, -1]

#%%
input_data = np.random.rand(10, 3, 7, 7)
filter_h, filter_w = 3, 3
stride = 1
pad = 0
N, C, H, W = input_data.shape
print(f'N, C, H, W = {N}, {C}, {H}, {W}')
out_h = (H + 2*pad - filter_h) // stride + 1
out_w = (W + 2*pad - filter_w) // stride + 1
print(f'out_h = {out_h}; out_w = {out_w}')
img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
img
