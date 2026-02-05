from multiprocessing import Pool
from time import sleep
from tqdm import tqdm
import numpy as np

def func_zipped(args):
    A, B, C = args
    X = A * B + C
    Y = A + B * C
    sleep(1)
    return X, Y

def main():
    num_rep = 100
    dim = 400

    rs = np.random.RandomState(seed = 1000)
    A_it = [rs.random((dim, dim)) for _ in range(num_rep)]
    B_it = [rs.random((dim, dim)) for _ in range(num_rep)]
    C_it = [rs.random((dim, dim)) for _ in range(num_rep)]

    X_sum = np.zeros((dim, dim))
    Y_sum = np.zeros((dim, dim))

    with Pool(processes = 5) as pool, tqdm(total = num_rep) as pbar:
        for X, Y in pool.imap(func_zipped, zip(A_it, B_it, C_it)):
            X_sum += X
            Y_sum += Y
            pbar.update(1)
            pbar.refresh()
    
    print(np.mean(X_sum))
    print(np.mean(Y_sum))

if __name__ == "__main__":
    main()