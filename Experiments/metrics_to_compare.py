import ot
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

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

    W2_to_bary = np.sqrt(W2_pot(eval_samples, true_bary_samples)) if true_bary_samples is not None else None

    return V_value, W2_to_bary

def evaluate_MC(approx_bary_it, input_measure_samples_collection_it, true_bary_samples_it, MC_size,
                num_parallel_process = None, pbar_text = None):
    '''
    Evaluate a computed approximate barycenter measure via Monte Carlo
    
    :param approx_bary_it: iterator returning samples from the computed approximate barycenter measure
    :param input_measure_samples_collection_it: iterator returning collections of samples from the input measures
    :param true_bary_samples_it: iterator returning samples from the true barycenter measure
    :param MC_size: number of Monte Carlo repetitions
    :param num_parallel_process: number of parallel processes for evaluation; if None, then evaluation is done without multiprocessing
    :param pbar_text: string displayed in the progress bar
    '''
    V_values_list = []
    W2_to_bary_list = []

    if num_parallel_process is not None:
        with Pool(processes = 5) as pool, tqdm(total = MC_size, desc = pbar_text) as pbar:
            for V_value, W2_to_bary in pool.imap(evaluate_zipped, 
                                    zip(approx_bary_it, 
                                        input_measure_samples_collection_it, 
                                        true_bary_samples_it)):
                V_values_list.append(V_value)
                W2_to_bary_list.append(W2_to_bary)
                pbar.update(1)
                pbar.refresh()
    else:
        for args in tqdm(zip(approx_bary_it, 
                             input_measure_samples_collection_it, 
                             true_bary_samples_it), 
                             total = MC_size, desc = pbar_text):
            V_value, W2_to_bary = evaluate_zipped(args)
            V_values_list.append(V_value)
            W2_to_bary_list.append(W2_to_bary)
    
    return V_values_list, W2_to_bary_list