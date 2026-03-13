import pandas as pd
import numpy as np
import statsmodels.api as sm
import zipfile
from sklearn.preprocessing import StandardScaler




def main():
    with zipfile.ZipFile("weather_forecasts.csv.zip", "r") as z:
        with z.open("weather_forecasts.csv") as f:
            df = pd.read_csv(f)
    x = [
        "high_or_low",
        "forecast_hours_before",
        "observed_temp",
        "forecast_temp",
        "observed_precip"]
    y = ["forecast_outlook"]
    df = df.sample(n=5000, random_state=42)
    df_x = df[x].copy()
    df_y = df[y].copy()

    # Drop NaN rows
    mask_nan = df_x.notna().all(axis=1) & df_y.notna().all(axis=1)
    df_x = df_x[mask_nan]
    df_y = df_y[mask_nan]

    # Binary encode high_or_low
    df_x["high_or_low"] = df_x["high_or_low"].map({"high": 1, "low": 0})

    numeric_cols = ["forecast_hours_before", "observed_temp", "forecast_temp", "observed_precip"]
    scaler = StandardScaler()
    df_x[numeric_cols] = scaler.fit_transform(df_x[numeric_cols])

    # Integer encode y (MNLogit needs integer labels, not one-hot)
    labels = df_y["forecast_outlook"].unique()
    label_to_int = {label: i for i, label in enumerate(labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    print(int_to_label)
    df_y["forecast_outlook"] = df_y["forecast_outlook"].astype("category").cat.codes

    x_train = sm.add_constant(df_x)
    y_train = df_y["forecast_outlook"]
    print("Training model...")

    model = sm.MNLogit(y_train, x_train)
    result = model.fit_regularized(alpha=0.1, method="l1", maxiter=30000, disp=True)

    print(result.summary())

if __name__ == "__main__":
    main()