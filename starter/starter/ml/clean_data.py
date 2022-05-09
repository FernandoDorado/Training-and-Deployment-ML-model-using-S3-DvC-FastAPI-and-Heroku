import pandas as pd
import numpy as np

# Load dataframe
df = pd.read_csv('../data/raw/census.csv')

# Save CSV
df.to_csv('../data/processed/census_processed.csv', index=False)