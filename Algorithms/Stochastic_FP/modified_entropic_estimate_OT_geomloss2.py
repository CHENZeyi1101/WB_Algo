from geomloss import SamplesLoss
import torch

class modified_entropic_OT_map_estimate_geomloss2:

    r'''
    Python class for constructing the regularized entropic OT map estimator
    Attributes: 
    X: numpy array, shape (n, d)
        Support of the empirical measure \widehat{\mu}; i.e., samples from the source distribution \mu \in \CP(\CX)
    Y: numpy array, shape (m, d)
        Support of the empirical measure \widehat{\nu}; i.e., samples from the input distribution \nu \in \CP(\CY)
    log: boolean, default True
    
    Methods:
    get_dual_potential(epsilon)
        Compute the dual potential g of the entropic regularized OT problem
    compute_modified_entropic_OT_map(X_new, radius)
        Compute the image of the points in the rows of X_new under the computed modified entropic OT map estimator
    '''
    
    def __init__(self, X, Y, log = None):
        self.X = torch.asarray(X)
        self.Y = torch.asarray(Y)
        self.g_potential = None
        self.epsilon = None
    
    def get_dual_potential(self, epsilon):
        '''
        In geomloss, the default cost function is the squared Euclidean distance with the factor 0.5.
        To maintain the consistency with the ott-based algorithm, we divide the regularization parameter by 2 and multiply the computed potential by 2.
        '''

        loss = SamplesLoss(loss = "sinkhorn", p = 2, 
                           blur = epsilon / 2, scaling = 0.999, 
                           truncate = 3, debias = False, potentials = True)
        _, g = loss(self.X, self.Y)

        self.g_potential = torch.squeeze(g) * 2
        self.epsilon = epsilon
    
    def compute_modified_entropic_OT_map(self, X_new, radius):
        X_new = torch.asarray(X_new)
        unmod_list = []
        chunk_size = 10**9 // self.Y.shape[0] 

        for X_new_sub in [X_new[i:min(i + chunk_size, X_new.shape[0])] for i in range(0, X_new.shape[0], chunk_size)]:
            unmod_list.append(self._compute_modified_entropic_OT_map_inner(X_new_sub))

        X_new_norm_halfsq_diff = 0.5 * (torch.sum(torch.square(X_new), dim = 1) - radius ** 2)
        modification_weights = torch.where(X_new_norm_halfsq_diff > 0, torch.exp(-1 / X_new_norm_halfsq_diff), 0.0)
        return (torch.vstack(unmod_list) + modification_weights[:, torch.newaxis] * X_new).numpy()
    
    def _compute_modified_entropic_OT_map_inner(self, X_new):
        diff_mat = (self.g_potential[torch.newaxis, :] - torch.square(torch.cdist(X_new, self.Y, p = 2))) / self.epsilon
        return torch.softmax(diff_mat, dim = 1) @ self.Y

