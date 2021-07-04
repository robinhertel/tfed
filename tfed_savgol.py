import numpy as np

def savgol(y, window_size, poly_order, deriv=0, weighted=False):
    # how many data points to consider to the left and right
    half_window= int((window_size -1 )/2)
    # compute convolution coefficients
    J = np.mat([[k**i for i in range(poly_order+1)] \
                 for k in range(-half_window, half_window+1)])
    if not weighted:
        W = np.eye(J.shape[0])
    else:
        xjxi = np.arange(-half_window,half_window+1)
        xjxidi3 = (np.abs(xjxi/(half_window)))**3
        wi = (1 - xjxidi3)**3
        W = np.diagflat(wi)
    JW = np.matmul(J.T,W)
    JWJ = np.linalg.inv(np.matmul(JW,J))
    m = np.matmul(JWJ,JW).A[deriv]
    # extrapolation
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    # convolve and return results
    return(np.convolve( m, y, mode='valid'))