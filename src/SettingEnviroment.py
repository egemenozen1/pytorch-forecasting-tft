
import pandas as pd

data = pd.read_csv('../data/AirQualityUCI.csv', sep=';')  # Noktalı virgül ayrımı varsa ona göre ayarlayabilirsin
print(data.head())

import numpy as np