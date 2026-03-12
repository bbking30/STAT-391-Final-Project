import zipfile
import pandas as pd
# from logistic import *
from preprocess import *

# Collecting Dataset from ZipFile
with zipfile.ZipFile("weather_forecasts.csv.zip") as z:
    weather_data = pd.read_csv(z.open("weather_forecasts.csv"))

weather_data = preprocess(weather_data)
