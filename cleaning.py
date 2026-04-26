# ================================
# DATA CLEANING & PREPROCESSING
# ================================

import pandas as pd
import numpy as np

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = pd.read_csv("customers-10000.csv")

print("Initial Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nData Info:\n")
print(df.info())

# -------------------------------
# STEP 2: REMOVE DUPLICATES
# -------------------------------
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# -------------------------------
# STEP 3: DROP IRRELEVANT COLUMNS
# -------------------------------
df = df.drop(columns=["notes", "temp_id"], errors='ignore')

# -------------------------------
# STEP 4: RENAME COLUMNS
# -------------------------------
df = df.rename(columns={
    "fname": "first_name",
    "lname": "last_name"
})

# -------------------------------
# STEP 5: HANDLE MISSING VALUES
# -------------------------------
print("\nMissing Values Before:\n", df.isna().sum())

# Fill numerical columns with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop rows where critical column missing
if "customer_id" in df.columns:
    df = df.dropna(subset=["customer_id"])

print("\nMissing Values After:\n", df.isna().sum())

# -------------------------------
# STEP 6: DATA TYPE CORRECTION
# -------------------------------
# Convert date columns
for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Convert numeric strings to numbers
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(",", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors='ignore')

# -------------------------------
# STEP 7: FORMAT STANDARDIZATION
# -------------------------------
# Clean text columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.lower().str.strip()

# Fix common categorical issues
if "gender" in df.columns:
    df["gender"] = df["gender"].replace({
        "m": "male",
        "f": "female"
    })

if "region" in df.columns:
    df["region"] = df["region"].replace({
        "west": "western",
        "south": "southern",
        "north": "northern",
        "east": "eastern"
    })

# -------------------------------
# STEP 8: OUTLIER HANDLING
# -------------------------------
for col in num_cols:
    if col in df.columns:
        upper_limit = df[col].quantile(0.99)
        df = df[df[col] <= upper_limit]

# -------------------------------
# STEP 9: FINAL VALIDATION
# -------------------------------
print("\nFinal Shape:", df.shape)

print("\nFinal Missing Values:\n", df.isna().sum())

assert df.duplicated().sum() == 0, "Duplicates still exist!"

# -------------------------------
# STEP 10: SAVE CLEAN DATA
# -------------------------------
df.to_csv("cleaned_customers.csv", index=False)

print("\n✅ Data Cleaning Completed Successfully!")