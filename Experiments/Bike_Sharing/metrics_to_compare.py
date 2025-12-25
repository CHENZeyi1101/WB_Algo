import ot
import numpy as np
from tqdm import tqdm

def W2_pot(X, Y): 
    r'''
    Compute the squared Wasserstein-2 distance between two empirical measures (using the POT library)
    '''
    M = ot.dist(X, Y)
    a, b = np.ones((X.shape[0],)) / X.shape[0], np.ones((Y.shape[0],)) / Y.shape[0]
    W2_sq = ot.emd2(a, b, M, numItermax=1e6)
    return W2_sq

def V_value_compute(self, bary_samples, input_samples_collection: dict):
        '''
        bary_samples denotes the samples from the true/approximated barycenter measure
        input_samples_collection is a dictionary with k keys, each key corresponds to the samples from the k-th input measure.
        '''
        V_value = 0
        for measure_index in tqdm(range(self.num_measures), desc = "V-value computation"):
            input_samples = np.array(input_samples_collection[measure_index])
            V_value += W2_pot(input_samples, bary_samples)
        V_value /= self.num_measures
        return V_value
    
def W2_to_bary_compute(self, bary_samples, generated_samples):
    '''
    Compute the (empirical) Wasserstein distance between the generated samples from the G-mapping
    and the barycenter samples at each iteration;
    '''
    W2_sq = W2_pot(generated_samples, bary_samples)
    return W2_sq