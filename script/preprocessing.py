# preprocessing.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load Data
# ----------------------------
def load_data(path):
    return pd.read_csv(path)

# ----------------------------
# Handle Missing Values
# ----------------------------
def handle_missing_values(df):
    # Fill Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # Fill Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df

# ----------------------------
# Encode Categorical Variables
# ----------------------------
def encode_categorical(df):
    # Encode Sex: male=0, female=1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # One-hot encode Embarked (drop first to avoid dummy trap)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    return df

# ----------------------------
# Keep Only Relevant Columns
# ----------------------------
def select_columns(df):
    return df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]]

# ----------------------------
# Visualize Outliers
# ----------------------------
def visualize_outliers(df, save=False):
    os.makedirs("../outputs", exist_ok=True)
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        plt.figure(figsize=(6, 4))
        df.boxplot(column=col)
        plt.title(f"Boxplot of {col}")
        if save:
            plt.savefig(f"../outputs/{col}_boxplot.png")
        plt.close()

# ----------------------------
# Preprocess Pipeline
# ----------------------------
def preprocess_pipeline(input_path, output_path):
    # Ensure outputs folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load data
    df = load_data(input_path)
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Encode categorical columns
    df = encode_categorical(df)

    # Step 4: Visualize outliers
    visualize_outliers(df, save=True)

    # Step 5: Select relevant columns for ML
    df = select_columns(df)

    # Step 6: Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\n✅ Preprocessing completed. Cleaned file saved at: {output_path}")

    return df
