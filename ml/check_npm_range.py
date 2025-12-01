# check_npm_range.py
from connect import load_financial_data
import pandas as pd
import numpy as np

df = load_financial_data()

# Drop missing
df = df.dropna()

# Winsorizing
lower = df["NPM"].quantile(0.05)
upper = df["NPM"].quantile(0.95)
df["NPM_winsor"] = np.clip(df["NPM"], lower, upper)

print("NPM ASLI (sebelum winsor & scaling):")
print(df["NPM"].describe())
print(f"Min: {df['NPM'].min():.4f}")
print(f"Max: {df['NPM'].max():.4f}")
print(f"Mean: {df['NPM'].mean():.4f}")

print("\n" + "="*50)
print("\nNPM_WINSOR (setelah winsor, sebelum scaling):")
print(df["NPM_winsor"].describe())
print(f"Min: {df['NPM_winsor'].min():.4f}")
print(f"Max: {df['NPM_winsor'].max():.4f}")
print(f"Mean: {df['NPM_winsor'].mean():.4f}")