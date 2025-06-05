# ============================================================
# NDVI Change-Point Detection Using Bayesian Change-Point Model (BCP)
# ============================================================

# This script processes NDVI time-series data to detect structural shifts
# in vegetation trends using Bayesian change-point analysis.
# It automates the retrieval, processing, and analysis of multiple CSV files.

# ============================================================
# 1. Load Required Libraries
# ============================================================

import os
import pandas as pd
import numpy as np
import ruptures as rpt  # Bayesian Change-Point Detection in Python

# ============================================================
# 2. Retrieve NDVI Data Files
# ============================================================

# Set working directory (modify as needed)
data_dir = "C:/Users/pa589/NAU/TREE_STRESS/TreeStress_detection/MODELS/Synthetic_datasets"

# Retrieve CSV files that start with "NDVI"
file_list = [f for f in os.listdir(data_dir) if f.startswith("NDVI") and f.endswith(".csv")]

# ============================================================
# 3. Preprocessing NDVI Values
# ============================================================

def preprocess_ndvi(ndvi_series):
    """Replace zero values with NaN and interpolate missing values."""
    ndvi_series = ndvi_series.replace(0, np.nan)  # Convert zero values to NaN
    ndvi_series = ndvi_series.interpolate(method='linear', limit_direction='both')  # Linear interpolation
    return ndvi_series

# ============================================================
# 4. Change-Point Detection Using Bayesian Change-Point Model
# ============================================================

def fit_bcp_model(data):
    """Apply Bayesian Change-Point analysis to detect NDVI shifts."""
    if "ndvi" not in data.columns or "day" not in data.columns:
        return {"change_point": -999}

    # Preprocess NDVI values
    data["ndvi"] = preprocess_ndvi(data["ndvi"])

    # Apply Bayesian Change-Point Model using PELT from 'ruptures'
    algo = rpt.Pelt(model="rbf").fit(data["ndvi"].values) # The PELT (Pruned Exact Linear Time) algorithm from the ruptures package is used to find structural shifts in NDVI trends.
    # The model type "rbf" (Radial Basis Function) is used, which is well-suited for detecting change points in noisy signals.
    breakpoints = algo.predict(pen=5)  # Adjust penalty (sensitivity) for better detection. Higher values make conservative detections, lower values allow more frequently detections.

    # Handle case where no change points are detected
    if not breakpoints:
        return {"change_point": -999}

    # Handle case where multiple change points are detected
    change_points = data["day"].iloc[breakpoints[:-1]]  # Exclude last point as 'ruptures' includes an endpoint
    result = {"num_change_points": len(change_points)}

    for i, cp in enumerate(change_points):
        result[f"change_point_{i+1}"] = cp

    return result
# ============================================================
# 5. Process Multiple NDVI Datasets
# ============================================================

results_list = []  # Store results

for file in file_list:
    file_path = os.path.join(data_dir, file)
    data = pd.read_csv(file_path)

    # Apply Bayesian change-point detection
    result = fit_bcp_model(data)
    result["file"] = file  # Add filename to results
    results_list.append(result)

# Convert results to a DataFrame
results_df = pd.DataFrame(results_list)

# ============================================================
# 6. Save the Results to a CSV File
# ============================================================

output_path = os.path.join(data_dir, "bcp_model", "results_bcp_model_Synthetic_datasets_folder.csv")
results_df.to_csv(output_path, index=False)

print(f"Results saved to: {output_path}")
print(results_df)
