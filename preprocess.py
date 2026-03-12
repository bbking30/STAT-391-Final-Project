import pandas as pd
import os


def preprocess(input_file):
    df = input_file.copy()

    # Drop N/A values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # clean text columns
    text_cols = ["forecast_outlook", "high_or_low", "city", "state"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Create features
    df["temp_error"] = df["observed_temp"] - df["forecast_temp"]
    df["abs_error"] = df["temp_error"].abs()

    return df