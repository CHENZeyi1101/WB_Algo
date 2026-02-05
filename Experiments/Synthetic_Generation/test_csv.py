from Experiments.CSV_read import *
import numpy as np

def main():
    num_measures = 5
    input_sampler = csv_input_sampler_SyntheticGeneration('../../WB_data/Synthetic_Generation/dim2_data/InstanceTheta2000_toy/input_samples/csv_files', num_measures = num_measures, multiplication_factor=1)
    input_sampler.set_streamers()

    while True:
        num_rows = int(input("Enter the number of rows: "))

        if num_rows <= 0:
            break

        samp_dict = input_sampler.sample(num_rows)

        for k in range(num_measures):
            print(f"Measure {k} =>")
            print(np.vstack(samp_dict[k]))
            print("\n")


if __name__ == "__main__":
    main()