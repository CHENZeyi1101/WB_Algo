from typing import Optional
import warnings
import pdb
import time

import numpy as np

from geomloss import SamplesLoss
import torch


class modified_entropic_OT_map_estimate_geomloss:

    r'''
    Python class for constructing the regularized entropic OT map estimator
    Attributes: 
    X: numpy array, shape (n, d)
        Support of the empirical measure \widehat{\mu}; i.e., samples from the source distribution \mu \in \CP(\CX)
    Y: numpy array, shape (m, d)
        Support of the empirical measure \widehat{\nu}; i.e., samples from the input distribution \nu \in \CP(\CY)
    log: boolean, default True
        If True, the class will log the outputs
    
    Methods:
    get_dual_potential(epsilon = None)
        Compute the dual potential g of the entropic regularized OT problem
    construct_entropic_OT_map(x)
        Construct the entropic OT map at the point x, and compute the image of x under the entropic OT map
    regularize_entropic_OT_map(M, x)
        Regularize the entropic OT map at the point x to make the corresponding potential strongly convex
    '''
    
    def __init__(self, X, Y, log = True):
        self.X = X
        self.Y = Y
        self.log = log
        self.g_potential = None
        self.epsilon = None
        self.dual_potentials = None
    
    def get_dual_potential(self, epsilon = None):
        '''
        In ott, the default cost function is the squared Euclidean distance (without the 0.5 factor).
        In our paper, we follow the convention in OT literature by using the 0.5 factor in front of the squared Euclidean distance, therefore the corresponding potential functions differ by a factor of 2.
        '''

        X, Y = torch.asarray(self.X), torch.asarray(self.Y)
        loss = SamplesLoss(loss = "sinkhorn", p = 2, 
                           blur = epsilon / 2, scaling = 0.999, 
                           truncate = 3, debias = False, potentials = True)
        f, g = loss(X, Y)
        f = np.squeeze(f.numpy()) * 2
        g = np.squeeze(g.numpy()) * 2

        self.dual_potentials = [f, g]
        self.g_potential = g
        self.epsilon = epsilon
    
    def construct_entropic_OT_map(self, x):
        Y = self.Y
        n = Y.shape[0]
        epsilon = self.epsilon
        g_potential = self.g_potential

        x_tile = np.tile(x, (n, 1))
        exponent_vec = (g_potential - np.sum(np.square(x_tile - Y), axis = 1)) / epsilon
        exponent_vec_max = np.max(exponent_vec)
        exponent_vec -= exponent_vec_max # normalize the exponent_vec for numerical stability in np.exp()

        # Convert warnings to exceptions within this block
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                numerator = Y.T @ np.exp(exponent_vec)
                denominator = np.sum(np.exp(exponent_vec))
                entropic_image = numerator / denominator
            except Warning as w:
                print(f"Warning converted to exception: {w}")
                pdb.set_trace()  # Trigger breakpoint for debugging
            except Exception as e:
                print(f"Error encountered: {e}")
                pdb.set_trace()  # Trigger breakpoint for debugging

        return entropic_image
    
    def regularize_entropic_OT_map(self, M, x):
        # Regularize the entropic OT map at point x to make the corresponding potential strongly convex
        # To avoid amendation of the original entropic OT map on the support of \widehat{\mu}
        # We set M = 0.5 * R^2, where R is the radius of the support of \widehat{\mu}

        entropic_image = self.construct_entropic_OT_map(x)
        half_xsq = 0.5 * x.T @ x
        if half_xsq <= M:
            return entropic_image
        else:
            regularized_entropic_image = entropic_image + np.exp(-1/(half_xsq - M)) * x
            return regularized_entropic_image
        


    
