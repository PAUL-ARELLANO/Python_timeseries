# ============================================================
# Python Script for NDVI Change-Point Detection Using PyMC3
# ============================================================
# Author: Paul Arellano, Ph.D.
# Date: Apr 30, 2025
# Description:
# This script analyzes synthetic NDVI datasets using Bayesian 
# change-point modeling with PyMC3. It identifies structural 
# shifts in vegetation trends and processes multiple CSV files 
# efficiently.
#
# Key Steps:
# 1. Load required libraries (pymc3, pandas, numpy, tqdm).
# 2. Retrieve the list of NDVI CSV files from the specified directory.
# 3. Define a function to preprocess NDVI values:
#     - Convert zero values to NaN.
#     - Apply linear interpolation to fill missing data.
# 4. Construct the Bayesian change-point model:
#     - Define priors for change-point, intercepts, and slope.
#     - Implement a switch logic for before/after the change-point.
#     - Define the likelihood function based on NDVI observations.
# 5. Loop through all datasets, apply preprocessing, and fit the model.
# 6. Extract the mean estimated change-point value (`cp_1_mean`).
# 7. Save results to a summary CSV file.
#
# Output:
# - A CSV file containing change-point estimates for each dataset.
#
# Notes:
# - Adjust file paths as needed.
# - Ensure the `pymc3` package is installed for Bayesian modeling.
# - Uses a single processing core for inference; modify accordingly.
#
# ============================================================

import pymc3 as pm
import pandas as pd
import numpy as np
import os
from glob import glob
import tqdm

# Set the directory containing the .csv files
directory = "C:/Users/pa589/NAU/TREE_STRESS/TreeStress_detection/MODELS/Synthetic_datasets"  # This path must be changed
file_list = glob(os.path.join(directory, "*.csv"))

# Function to preprocess NDVI values
def preprocess_ndvi(ndvi_series):
    """
    Interpolates NA and zero values by interpolating the neighbors values.
    """
    ndvi_series = ndvi_series.copy()
    
    # Identify positions where values are NA or zero
    mask = (ndvi_series.isna()) | (ndvi_series == 0)
    
    # Replace with interpolated values
    ndvi_series[mask] = ndvi_series.interpolate(method='linear', limit_direction='both')
    
    return ndvi_series

# Define a function to fit a change-point model to a dataset
def fit_change_point(file):
    # Read the CSV file
    ndvi_df = pd.read_csv(file)
    
    # Check if the required columns are present
    if "ndvi" not in ndvi_df.columns or "day" not in ndvi_df.columns:
        print(f"Skipping {file}, required columns ('ndvi' and 'day') are missing.")
        return {"file_name": os.path.basename(file), "cp_1_mean": np.nan}
    
    # Extract and preprocess NDVI values
    day = ndvi_df["day"].values
    ndvi = preprocess_ndvi(ndvi_df["ndvi"])

    # Define the change-point model
    with pm.Model() as model:
        # Priors
        cp = pm.Uniform("cp", lower=min(day), upper=max(day))  # Change-point
        intercept_1 = pm.Normal("intercept_1", mu=0, sigma=10)  # Before change-point
        intercept_2 = pm.Normal("intercept_2", mu=0, sigma=10)  # After change-point
        
        # Switch point logic
        slope = pm.Normal("slope", mu=0, sigma=1)
        ndvi_hat = pm.math.switch(
            day < cp,
            intercept_1 + slope * day,
            intercept_2 + slope * day
        )
        
        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=1)
        likelihood = pm.Normal("ndvi_obs", mu=ndvi_hat, sigma=sigma, observed=ndvi)
        
        # Inference
        trace = pm.sample(1000, tune=1000, target_accept=0.95, cores=1, progressbar=False)
    
    # Extract the mean value of the change point (cp)
    cp_mean = trace["cp"].mean()
    
    return {"file_name": os.path.basename(file), "cp_1_mean": cp_mean}

# Loop through all files and fit the model
results = []
for file in tqdm.tqdm(file_list, desc="Processing files"):
    result = fit_change_point(file)
    results.append(result)

# Save the results to a DataFrame
results_df = pd.DataFrame(results)

# Print and save the results
print(results_df)
results_df.to_csv("results_summary.csv", index=False)
