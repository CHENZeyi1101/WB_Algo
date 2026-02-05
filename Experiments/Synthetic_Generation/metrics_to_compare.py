import ot
import numpy as np
from tqdm import tqdm

def W2_pot(X, Y): 
    r'''
    Compute the squared Wasserstein-2 distance between two empirical measures (using the POT library)
    '''
    M = ot.dist(X, Y)
    a, b = np.ones((X.shape[0],)) / X.shape[0], np.ones((Y.shape[0],)) / Y.shape[0]
    W2_sq = ot.emd2(a, b, M, numItermax=1e7)
    return W2_sq

def V_value_compute(bary_samples, input_samples_collection: dict):
        '''
        bary_samples denotes the samples from the true/approximated barycenter measure
        input_samples_collection is a dictionary with k keys, each key corresponds to the samples from the k-th input measure.
        '''
        V_value = 0
        for measure_index in tqdm(range(len(input_samples_collection)), desc = "V-value computation"):
            input_samples = np.array(input_samples_collection[measure_index])
            V_value += W2_pot(input_samples, bary_samples)
        V_value /= len(input_samples_collection)
        return V_value
    
def W2_to_bary_compute(bary_samples, generated_samples):
    '''
    Compute the (empirical) Wasserstein distance between the generated samples from the G-mapping
    and the barycenter samples at each iteration;
    '''
    W2_sq = W2_pot(generated_samples, bary_samples)
    return W2_sq

def evaluate_zipped(args):
    '''
    Compute both the (empirical) V-value and the (empirical) Wasserstein distance (without square) between samples from an approximate barycenter and
    and samples from the true barycenter.
    The inputs are passed via a tuple for ease of parallelization
    '''
    eval_samples, input_measure_samples_collection, true_bary_samples = args

    V_value = 0
    for measure_index in range(len(input_measure_samples_collection)):
        V_value += W2_pot(eval_samples, np.array(input_measure_samples_collection[measure_index]))
    V_value /= len(input_measure_samples_collection)

    W2_to_bary = np.sqrt(W2_pot(eval_samples, true_bary_samples))

    return V_value, W2_to_bary