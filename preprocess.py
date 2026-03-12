import pandas as pd
import os


def preprocess_weather_data(input_file, output_file="data/weather_cleaned.csv"):

    # create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(input_file)

    columns = [
        "forecast_hours_before",
        "forecast_temp",
        "observed_temp",
        "forecast_outlook",
        "high_or_low",
        "city",
        "state"
    ]

    df = df[columns]

    # remove null rows
    df = df.dropna()

    # clean text columns
    text_cols = ["forecast_outlook", "high_or_low", "city", "state"]

    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # remove duplicates
    df = df.drop_duplicates()

    # create features
    df["temp_error"] = df["observed_temp"] - df["forecast_temp"]
    df["abs_error"] = df["temp_error"].abs()

    # save to data folder
    df.to_csv(output_file, index=False)

    print(f"Saved cleaned dataset to {output_file}")


if __name__ == "__main__":
    preprocess_weather_data("data/weather_forecasts.csv")
