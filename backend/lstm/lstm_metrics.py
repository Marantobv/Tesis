import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Cargar el archivo CSV
df = pd.read_csv('SPX_2.csv')

# Seleccionar las columnas que deseas utilizar como características
features = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values

# Escalar los datos
scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_close = MinMaxScaler(feature_range=(-1, 1))

scaled_features = scaler_features.fit_transform(features)
scaled_close = scaler_close.fit_transform(df[['Close']].values)

# Convertir los datos a tensores de PyTorch y crear las secuencias
def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

SEQ_LENGTH = 60  # Ventana de 30 días
X, y = create_sequences(scaled_features, scaled_close, SEQ_LENGTH)

# Separar los datos en entrenamiento y validación
train_size = int(0.95 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Crear DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Inicializar el modelo
input_size = X.shape[2]  # El número de características (6: Open, High, Low, Close, Adj Close, Volume)
hidden_size = 64
num_layers = 2
output_size = 1  # Predicción de un solo valor (el precio de cierre)

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Entrenamiento del modelo
for epoch in range(EPOCHS):
    model.train()
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

# Evaluar en el conjunto de validación y calcular métricas
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        y_true.append(labels.cpu().numpy())
        y_pred.append(outputs.cpu().numpy())
    
    # Convertir las listas a arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Invertir la escala SOLO para los valores de 'Close'
    y_true = scaler_close.inverse_transform(y_true)
    y_pred = scaler_close.inverse_transform(y_pred)
    
    # Calcular MAE, RMSE y MAPE
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

# Predicción para el próximo día
input_sequence = scaled_features[-SEQ_LENGTH:]
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

# Predicción
with torch.no_grad():
    prediction = model(input_sequence)

# Invertir la escala SOLO para el valor de 'Close'
predicted_close_price = scaler_close.inverse_transform(prediction.cpu().numpy())

print(f"Predicción del precio de cierre para el próximo día: {predicted_close_price[0][0]}")
