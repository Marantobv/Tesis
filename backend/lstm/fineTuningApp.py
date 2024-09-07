import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch
import numpy as np
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
df['Date'] = pd.to_datetime(df['Date'])

# Split data into pre-2021 and post-2021
pre_2021 = df[df['Date'] < '2021-01-01']
post_2021 = df[df['Date'] >= '2021-01-01']

# Preprocess pre-2021 data
pre_2021_features = pre_2021[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values

scaler_features = MinMaxScaler(feature_range=(-1, 1))
scaler_close = MinMaxScaler(feature_range=(-1, 1))

scaled_pre_2021_features = scaler_features.fit_transform(pre_2021_features)
scaled_pre_2021_close = scaler_close.fit_transform(pre_2021[['Close']].values)

# Preprocess post-2021 data
post_2021_features = post_2021[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values

scaler_features_post_2021 = MinMaxScaler(feature_range=(-1, 1))
scaled_post_2021_features = scaler_features_post_2021.fit_transform(post_2021_features)
scaled_post_2021_close = scaler_close.transform(post_2021[['Close']].values)

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

SEQ_LENGTH = 30  # Ventana de 60 días
X_pre_2021, y_pre_2021 = create_sequences(scaled_pre_2021_features, scaled_pre_2021_close, SEQ_LENGTH)
X_post_2021, y_post_2021 = create_sequences(scaled_post_2021_features, scaled_post_2021_close, SEQ_LENGTH)

# Create DataLoaders
train_loader_pre_2021 = DataLoader(TensorDataset(X_pre_2021, y_pre_2021), batch_size=32, shuffle=True)
train_loader_post_2021 = DataLoader(TensorDataset(X_post_2021, y_post_2021), batch_size=32, shuffle=True)

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
input_size_pre_2021 = X_pre_2021.shape[2]  # El número de características (6: Open, High, Low, Close, Adj Close, Volume)
input_size_post_2021 = X_post_2021.shape[2]  # El número de características (7: Open, High, Low, Close, Adj Close, Volume, Sentiment)
hidden_size = 64
num_layers = 2
output_size = 1  # Predicción de un solo valor (el precio de cierre)

model = LSTMModel(input_size_pre_2021, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS_PRE_2021 = 50
EPOCHS_POST_2021 = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Training function
def train_model(model, train_loader, epochs, criterion, optimizer, device):
    final_loss = 0
    for epoch in range(epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        final_loss = loss.item()
    return final_loss

# Train on pre-2021 data
print("Training on pre-2021 data:")
train_model(model, train_loader_pre_2021, EPOCHS_PRE_2021, criterion, optimizer, device)

# Fine-tune on post-2021 data
print("\nFine-tuning on post-2021 data:")
model.lstm = nn.LSTM(input_size_post_2021, hidden_size, num_layers, batch_first=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate for fine-tuning

final_loss = train_model(model, train_loader_post_2021, EPOCHS_POST_2021, criterion, optimizer, device)
print(f"Final training loss: {final_loss:.4f}")

# Make prediction
input_sequence = scaled_post_2021_features[-SEQ_LENGTH:]
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    prediction = model(input_sequence)

# Invertir la escala SOLO para el valor de 'Close'
predicted_close_price = scaler_close.inverse_transform(prediction.cpu().numpy())

print(f"Predicted closing price for the next day: {predicted_close_price[0][0]}")

