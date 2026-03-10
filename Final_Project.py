import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm


weather_data = pd.read_csv("weather_forecasts.csv")
weather_data.dropna(inplace=True)
print(weather_data.head())
