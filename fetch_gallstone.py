from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

os.makedirs('figures', exist_ok=True)


print(f"Python version: {sys.version}")
print(f"pandas version: {pd.__version__}")

try:
    print("Fetching Gallstone dataset from UCI repository...")
    gallstone = fetch_ucirepo(id=1150)
    
    print("\nDataset metadata:")
    print(gallstone.metadata)
    
    X = gallstone.data.features 
    y = gallstone.data.targets
    

    data = pd.concat([X, y], axis=1)
    
    # Print basic information about the dataset
    print(f"\nDataset loaded successfully with shape: {data.shape}")
    print("\nFeatures:")
    for col in X.columns:
        print(f"- {col}")
    print("\nTarget:")
    print(f"- {y.columns[0]}")
    
    # Check the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Save the dataset to a clean Excel file for future use
    data.to_excel('gallstone_clean.xlsx', index=False)
    print("\nClean dataset saved to 'gallstone_clean.xlsx'")
    
except Exception as e:
    print(f"Error fetching dataset: {e}")
    import traceback
    traceback.print_exc()
