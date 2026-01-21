import pandas as pd
from tqdm import tqdm

csv_dir = "../../WB_data/Bike_Sharing"

num_measures = 5
all_summaries = []

df_full = pd.read_csv(f"{csv_dir}/posterior_full.csv", 
                 usecols=range(7, 16), 
                 skiprows=52,
                 header=None)

summary_full = pd.DataFrame({
    "mean": df_full.mean(),
    "variance": df_full.var(),
    "std": df_full.std(),
    "min": df_full.min(),
    "max": df_full.max(),
    "range": df_full.max() - df_full.min(),
    "median": df_full.median(),
    "q25": df_full.quantile(0.25),
    "q75": df_full.quantile(0.75)
})

# ---- full posterior ----
summary_full["source"] = "full"
summary_full["col_id"] = summary_full.index
all_summaries.append(summary_full)

for i in tqdm(range(num_measures), desc="Processing split posteriors"):
    df_split = pd.read_csv(f"{csv_dir}/posterior_split_{i}.csv",
                        usecols=range(7, 16), 
                        skiprows=52,
                        header=None)
    summary_split = pd.DataFrame({
        "mean": df_split.mean(),
        "variance": df_split.var(),
        "std": df_split.std(),
        "min": df_split.min(),
        "max": df_split.max(),
        "range": df_split.max() - df_split.min(),
        "median": df_split.median(),
        "q25": df_split.quantile(0.25),
        "q75": df_split.quantile(0.75)
    })

    summary_split["source"] = f"split_{i}"
    summary_split["col_id"] = summary_split.index
    all_summaries.append(summary_split)

summary_all = pd.concat(all_summaries, ignore_index=True)

summary_all.to_csv(
    f"{csv_dir}/posterior_summary_all.csv",
    index=False
)
print("Summary statistics saved to posterior_summary_all.csv")