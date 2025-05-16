import pandas as pd
import os

# Print current directory for verification
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir())
print("Files in gallstone-data:", os.listdir("./gallstone-data"))

# Try to load the downloaded Excel file
try:
    # Try with explicit engine
    file_path = './gallstone-data/dataset-uci.xlsx'
    print(f"Attempting to load: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    data = pd.read_excel(file_path, engine='openpyxl')
    print("\nDataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
    
    # Display column names
    print("\nColumns in the dataset:")
    for i, col in enumerate(data.columns):
        print(f"{i+1}. {col}")
    
    # Display first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Check for missing values
    missing = data.isnull().sum()
    print("\nMissing values:")
    print(missing[missing > 0] if missing.any() > 0 else "No missing values!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
