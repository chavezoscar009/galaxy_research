import numpy as np


def mask(row_min, row_max, data):


    '''
    This function will attempt to mask out the 2D arrays row-wise. This means that we will get rid of those pesky bad pixels we see in the
    2D spectrum. The way it will do that is by zeroing out everything outside of row_min and row_max.

    Parameters
    ------------------
    row_min: This is the minimum row index of the array that we will stop masking. ie, [0: row_min]
    row_max: this is the maximum row index where the splicing will begin to the end. ie [row_max, :]
    data: the data that will be masked outside row_min and row_max

    Output
    ------------------
    mask: this is the spliced array between row_min and row_max specified in the input of the function
    '''
    
    masked = np.ones(data.shape)

    for i in range(len(data[:, 0])):
        
        if i >= row_min and i <= row_max:
            continue

        else:
            masked[i, :] = np.zeros(len(data[0,:]))

    return masked
