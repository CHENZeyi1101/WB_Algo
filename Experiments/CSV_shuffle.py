import pandas as pd
from tqdm import tqdm

def csv_shuffle(csv_path, output_path, seed = 42):

    # read csv
    df = pd.read_csv(csv_path)

    # reshuffle rows
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # save to new csv
    df_shuffled.to_csv(output_path, index=False)



if __name__ == "__main__":
    num_measures = 5
    dim = 2
    for i in tqdm(range(num_measures), desc="Shuffling CSV files"):
        old_csv_path = f"../../WB_Data/input_measure_samples_{i}.csv"
        csv_evaluate_dir = f"../../WB_Data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files_InstanceTheta2000/for_evaluation"
        new_csv_path = f"{csv_evaluate_dir}/input_measure_samples_{i}_for_evaluation.csv"
        csv_shuffle(old_csv_path, new_csv_path, seed = 42)

    
