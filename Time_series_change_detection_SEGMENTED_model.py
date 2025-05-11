# ============================================================
# NDVI Change-Point Detection Using Segmented Regression (Python Version)
# ============================================================

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm
import statsmodels.formula.api as smf
import ruptures as rpt

# ============================================================
# 1. Retrieve NDVI Data Files
# ============================================================

# Set working directory (modify as needed)
data_dir = "C:/Users/pa589/NAU/TREE_STRESS/TreeStress_detection/MODELS/Synthetic_datasets"

# Retrieve CSV files that start with "NDVI"
file_list = [f for f in os.listdir(data_dir) if f.startswith("NDVI") and f.endswith(".csv")]

# ============================================================
# 2. Preprocessing NDVI Values
# ============================================================

def preprocess_ndvi(ndvi_series):
    """Replace zero values with NaN and interpolate missing values."""
    ndvi_series = ndvi_series.replace(0, np.nan)  # Convert zero values to NaN
    ndvi_series = ndvi_series.interpolate(method='linear', limit_direction='both')  # Linear interpolation
    return ndvi_series

# ============================================================
# 3. Change-Point Detection Using Segmented Regression
# ============================================================

def fit_segmented_model(data):
    """Fit a segmented regression model to detect structural shifts in NDVI."""
    if "ndvi" not in data.columns or "day" not in data.columns:
        return {"change_point": np.nan, "x_value": np.nan}

    # Preprocess NDVI values
    data["ndvi"] = preprocess_ndvi(data["ndvi"])

    # Fit a linear model
    model = smf.ols("ndvi ~ day", data=data).fit()

    # Use the 'ruptures' package to detect breakpoints
    algo = rpt.Pelt(model="rbf").fit(data["ndvi"].values)
    breakpoints = algo.predict(pen=5)  # Adjust penalty for better detection

    if not breakpoints or breakpoints[0] >= len(data["day"]):  # Ensure valid result
        return {"change_point": np.nan, "x_value": np.nan}

    change_point = data["day"].iloc[breakpoints[0]]
    return {"change_point": change_point, "x_value": change_point}

# ============================================================
# 4. Process Multiple NDVI Datasets
# ============================================================

results_list = []  # Store results

for file in file_list:
    file_path = os.path.join(data_dir, file)
    data = pd.read_csv(file_path)

    # Fit model and extract change points
    result = fit_segmented_model(data)
    result["file"] = file  # Add filename to results
    results_list.append(result)

# Convert results to a DataFrame
results_df = pd.DataFrame(results_list)

# ============================================================
# 5. Save the Results to a CSV File
# ============================================================

output_path = os.path.join(data_dir, "segmented_model", "results_segmented_model.csv")
results_df.to_csv(output_path, index=False)

print(f"Results saved to: {output_path}")
print(results_df)
