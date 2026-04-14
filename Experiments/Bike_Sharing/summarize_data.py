"""
Summarize bike_sharing_selected_features.csv and output a LaTeX summary table.

Two-panel layout:
  Panel A – Continuous variables: Mean, Std, Min, Median, Max
  Panel B – Categorical/binary variables: frequency distribution per category
"""

import pandas as pd
import os

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "bike_sharing_selected_features.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "data_summary.txt")

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=["intercept"], errors="ignore")
N  = len(df)

# ── variable classification ────────────────────────────────────────────────────
CONTINUOUS = ["cnt", "temp", "atemp", "hum", "windspeed"]

CONTINUOUS_META = {
    "cnt":       ("Count$^*$",           "Total bike rentals (response variable)"),
    "temp":      ("Temp.",              "Normalized temperature"),
    "atemp":     ("Feels-like Temp.",   "Normalized feeling temperature"),
    "hum":       ("Humidity",           "Normalized humidity"),
    "windspeed": ("Wind Speed",         "Normalized wind speed"),
}

# Each entry: (display_name, description, {value: label})
# For "hr", value=None signals special handling (range + mode).
CATEGORICAL_META = {
    "hr": (
        "Hour",
        "Hour of the day (ordinal, 0--23)",
        None,
    ),
    "season": (
        "Season",
        "Meteorological season",
        {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"},
    ),
    "workingday": (
        "Working Day",
        "Whether the day is a working day",
        {0: "No", 1: "Yes"},
    ),
    "weathersit": (
        "Weather",
        "Weather situation",
        {1: "Clear", 2: "Mist", 3: "Light rain/snow", 4: "Heavy rain/snow"},
    ),
}

# ── helper ─────────────────────────────────────────────────────────────────────
def fmt(x, decimals=3):
    if x == int(x):
        return str(int(x))
    return f"{x:.{decimals}f}"

# ── Panel A rows ───────────────────────────────────────────────────────────────
panel_a_rows = []
for col in CONTINUOUS:
    s = df[col]
    label, desc = CONTINUOUS_META[col]
    panel_a_rows.append((
        label, desc,
        fmt(s.mean()), fmt(s.std()),
        fmt(s.min()), fmt(s.median()), fmt(s.max()),
    ))

# ── Panel B rows ───────────────────────────────────────────────────────────────
panel_b_rows = []
for col in ["hr", "season", "workingday", "weathersit"]:
    label, desc, cat_labels = CATEGORICAL_META[col]
    if cat_labels is None:
        # Ordinal with many levels: show range and mode
        mode_val = int(df[col].mode()[0])
        dist_str = rf"Range: 0--23 \quad Mode: {mode_val}"
    else:
        counts = df[col].value_counts().sort_index()
        parts = []
        for val, cat_label in cat_labels.items():
            pct = 100.0 * counts.get(val, 0) / N
            parts.append(rf"{cat_label}: {pct:.1f}\%")
        dist_str = r" \quad ".join(parts)
    panel_b_rows.append((label, desc, dist_str))

# ── build LaTeX ────────────────────────────────────────────────────────────────
# Single table: 7 columns. Categorical rows span cols 3-7 with the distribution.
# col spec: Variable | Description | Mean | Std | Min | Median | Max
L = []

L.append(r"\begin{table}[ht]")
L.append(r"  \centering")
L.append(r"  \caption{Summary statistics of the bike-sharing dataset "
         rf"($N = {N:,}$).}}")
L.append(r"  \label{tab:bike_summary}")
L.append(r"  \begin{adjustbox}{max width=\textwidth}")
L.append(r"  \begin{tabular}{llrrrrr}")
L.append(r"    \toprule")
L.append(r"    \textbf{Variable} & \textbf{Description} & "
         r"\textbf{Mean} & \textbf{Std} & "
         r"\textbf{Min} & \textbf{Median} & \textbf{Max} \\")
L.append(r"    \midrule")

# Continuous section
L.append(r"    \multicolumn{7}{l}{\textit{Continuous variables}} \\")
L.append(r"    \midrule")
for label, desc, mean, std, mn, med, mx in panel_a_rows:
    L.append(f"    {label} & {desc} & {mean} & {std} & {mn} & {med} & {mx} \\\\")

L.append(r"    \midrule")

# Categorical section
L.append(r"    \multicolumn{7}{l}{\textit{Categorical \& binary variables}} \\")
L.append(r"    \midrule")
for label, desc, dist in panel_b_rows:
    L.append(rf"    {label} & {desc} & \multicolumn{{5}}{{l}}{{{dist}}} \\")

L.append(r"    \bottomrule")
L.append(r"  \end{tabular}")
L.append(r"  \end{adjustbox}")
L.append(r"\end{table}")

latex_str = "\n".join(L)

# ── write output ───────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    f.write("% ── Required packages in preamble ────────────────────────────────\n")
    f.write("% \\usepackage{booktabs}\n")
    f.write("% \\usepackage{adjustbox}\n")
    f.write("%\n")
    f.write(latex_str)
    f.write("\n")

print(f"Summary table written to: {OUT_PATH}")
print()
print(latex_str)
