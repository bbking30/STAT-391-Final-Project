import numpy as np
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm

# Preprocessing for dataset
zip_path = 'weather_forecasts.csv.zip'
with zipfile.ZipFile(zip_path, "r") as z:
    files = [f for f in z.namelist() if not f.startswith("__MACOSX")]

    with zipfile.ZipFile("clean_weather.zip", "w") as new_zip:
        for f in files:
            new_zip.writestr(f, z.read(f))

# Collected dataset / unzipped csv
weather_data = pd.read_csv('clean_weather.zip', compression='zip')
# Drop N/A values
weather_data.dropna(inplace=True)
