import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

# Fijar la semilla para reproducibilidad
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Cargar el archivo CSV
df = pd.read_csv('SP500_FullSentiment.csv')

# Seleccionar las columnas que deseas utilizar como características
features = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values

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

SEQ_LENGTH = 30  # Ventana de 30 días
X, y = create_sequences(scaled_features, scaled_close, SEQ_LENGTH)

# Dividir los datos en 80% entrenamiento y 20% prueba
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Crear DataLoader para los conjuntos de datos de entrenamiento y prueba
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

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
input_size = X.shape[2]  # El número de características (7: Open, High, Low, Close, Adj Close, Volume, Sentiment)
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

# Evaluación del modelo en el conjunto de prueba
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        predictions.append(outputs.cpu().numpy())
        actuals.append(labels.numpy())

# Convertir las listas a arrays
predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Desescalar las predicciones y los valores reales
predictions_descaled = scaler_close.inverse_transform(predictions)
actuals_descaled = scaler_close.inverse_transform(actuals)

# Calcular las métricas de error
mape = np.mean(np.abs((actuals_descaled - predictions_descaled) / actuals_descaled)) * 100
mae = mean_absolute_error(actuals_descaled, predictions_descaled)
rmse = np.sqrt(mean_squared_error(actuals_descaled, predictions_descaled))

print(f'MAPE: {mape:.4f}%')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Predicción para el próximo día
input_sequence = scaled_features[-SEQ_LENGTH:]
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(input_sequence)

# Invertir la escala para la predicción del próximo día
predicted_close_price = scaler_close.inverse_transform(prediction.cpu().numpy())
print(f"Predicción del precio de cierre para el próximo día: {predicted_close_price[0][0]}")
