import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv('SP500_FullSentimentReduced.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


df['DateFormat'] = pd.to_datetime(df['Date'])

df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

df['Volatility'] = df['Log_Return'].rolling(window=21).std() * np.sqrt(252)

df['Volatility'].fillna(0, inplace=True)

plt.figure(figsize=(14, 7))
plt.plot(df['DateFormat'], df['Volatility'], label='Volatilidad (Ventana de 21 días)', color='blue')
plt.title('Volatilidad Mensual del Índice S&P 500')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.legend()
plt.show()