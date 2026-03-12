import zipfile
from logistic import *
from preprocess import *

# Collecting Dataset from ZipFile
with zipfile.ZipFile("weather_forecasts.csv.zip") as z:
    weather_data = pd.read_csv(z.open("weather_forecasts.csv"))

# Drop N/A values across all rows
weather_data.dropna(inplace=True)
