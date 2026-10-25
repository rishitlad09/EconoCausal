import pandas as pd
from pathlib import Path

# Find the Excel dataset
file_path = Path("data/Online Retail - Copy.xlsx")

# Check whether the file exists
if file_path.exists():
    data = pd.read_excel(file_path)

    print("Dataset loaded successfully!")

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns.tolist())

else:
    print("Dataset not found!")
    print("Please check the data folder and file name.")