import numpy as np
import scipy
from scipy.special import logsumexp


def softmax(dxy, beta=1, axis=None):
    """:return
        axis = None: pXY propto exp(-beta * dxy)
        axis = 1: pY_X propto exp(-beta * dxy)
        axis = 0: pX_Y propto exp(-beta * dxy)
    """
    log_z = logsumexp(-beta * dxy, axis, keepdims=True)
    return np.exp(-beta * dxy - log_z)



def mi(p_xy):
    """ Calculate mutual information of a distribution P(x,y) 

    Input: 
    p_xy: An X x Y array giving p(x,y)
    
    Output:
    The mutual information I[X:Y], a nonnegative scalar,
    """
    p_x = p_xy.sum(axis=-1, keepdims=True)
    p_y = p_xy.sum(axis=-2, keepdims=True)
    return scipy.special.xlogy(p_xy, p_xy).sum() - scipy.special.xlogy(p_x, p_x).sum() - scipy.special.xlogy(p_y, p_y).sum()



def gnid(pW_X, pV_X, pX, unnormalized=False):

    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T

    pXW = pW_X * pX
    pWV = pXW.T.dot(pV_X)
    pWW = pXW.T.dot(pW_X)
    pVV = (pV_X * pX).T.dot(pV_X)

    if unnormalized:
        return mi(pWV)
    else:
        return 1 - mi(pWV) / (np.max([mi(pWW), mi(pVV)]))
