import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Print current directory and files for debugging
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())

# Load the dataset with explicit engine
try:
    data = pd.read_excel('dataset-uci.xlsx', engine='openpyxl')

    
    # Print basic information about the dataset
    print("\nDataset shape:", data.shape)
    print("\nColumn names:")
    for col in data.columns.tolist():
        print(f"- {col}")
    print("\nSample data (first 3 rows):")
    print(data.head(3))
    print("\nData types:")
    for col, dtype in data.dtypes.items():
        print(f"- {col}: {dtype}")
    print("\nCheck for missing values (showing only columns with missing values):")
    missing = data.isnull().sum()
    print(missing[missing > 0] if missing.any() > 0 else "No missing values!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
