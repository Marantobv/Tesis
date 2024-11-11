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

# df_train = df[df['Date'] < '2021-01-01']
# df_finetune = df[df['Date'] >= '2021-01-01']

df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

df['Signal'] = 0
df['Signal'][df['SMA_20'] > df['SMA_50']] = 1  # Ciclo alcista
df['Signal'][df['SMA_20'] < df['SMA_50']] = -1 # Ciclo bajista

plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Precio de Cierre', color='black')
plt.plot(df['Date'], df['SMA_20'], label='Media Móvil de 20 días', color='blue')
plt.plot(df['Date'], df['SMA_50'], label='Media Móvil de 50 días', color='red')
plt.fill_between(df['Date'], df['Close'].min(), df['Close'].max(), 
                 where=(df['Signal'] == 1), color='green', alpha=0.1, label='Ciclo Alcista')
plt.fill_between(df['Date'], df['Close'].min(), df['Close'].max(), 
                 where=(df['Signal'] == -1), color='red', alpha=0.1, label='Ciclo Bajista')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Muestra cada 6 meses, ajusta según prefieras
plt.xticks(rotation=45)  # Rota las etiquetas del eje X para que no se superpongan

plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.title('Tendencia de Ciclos Alcistas y Bajistas en el S&P 500')
plt.legend()
plt.tight_layout()  # Ajusta el gráfico para que las etiquetas no se corten
plt.show()