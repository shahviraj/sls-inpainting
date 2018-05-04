
import numpy as np
import scipy as sp
from vec import vec
import matplotlib.pyplot as plt


""" currently only support width (and height) * resize_ratio is an interger! """
def setup(input_img,x_shape):

    #spare = 0.25 * box_size
    print(input_img.dtype)
    #mask = np.ones(input_img.shape)
    mask = np.logical_not(input_img > 0.9)
    print(np.amin(input_img))
    print(np.amax(input_img))
    print(np.sum(mask))
    mask_not = input_img > 0.9
    #mask_not = (input_img == 255).all(axis=3)
    print(mask.shape)
    #mask3 = np.repeat(mask[:, :, :, np.newaxis], 3, axis=3)
    mask = mask.astype(np.uint8)
    mask_not = mask_not.astype(np.uint8)
	
    #for i in range(total_box):

    #    start_row = spare
    #    end_row = x_shape[1] - spare - box_size - 1
    #    start_col = spare
    #    end_col = x_shape[2] - spare - box_size - 1

    #    idx_row = int(np.random.rand(1) * (end_row - start_row) + start_row)
    #    idx_col = int(np.random.rand(1) * (end_col - start_col) + start_col)

    #    mask[0,idx_row:idx_row+box_size,idx_col:idx_col+box_size,:] = 0.


    def A_fun(x):
        y = np.multiply(x, mask)# + mask_not;
        return y

    def AT_fun(y):
        x = np.multiply(y, mask)# + mask_not;
        return x

    return (A_fun, AT_fun, mask, mask_not)


