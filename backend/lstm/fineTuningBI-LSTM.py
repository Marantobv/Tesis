#SUPUESTA CORRECCION PERO ME BAJA TODAS LAS PREDICCIONES QUE EN REALIDAD CONCUERDA CON EL MAPE
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# Configuración de semillas para reproducibilidad
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Preparar los datos
df = pd.read_csv('SP500_FullSentiment.csv')

# Dividir en conjunto de entrenamiento y fine-tuning
df_train = df[df['Date'] < '2021-01-01']
df_finetune = df[df['Date'] >= '2021-01-01']

# Further split the fine-tuning data into validation and test sets
validation_split = 0.8  # 80% for validation, 20% for test
split_index = int(len(df_finetune) * validation_split)

df_validation = df_finetune.iloc[:split_index]
df_test = df_finetune.iloc[split_index:]

# Prepare features and targets for each set
features_train = df_train[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
target_train = df_train[['Close']].values

features_validation = df_validation[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values
target_validation = df_validation[['Close']].values

features_test = df_test[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values
target_test = df_test[['Close']].values

# Escalar los datos
scaler_features_train = MinMaxScaler(feature_range=(-1, 1))
scaler_close_train = MinMaxScaler(feature_range=(-1, 1))

scaled_features_train = scaler_features_train.fit_transform(features_train)
scaled_close_train = scaler_close_train.fit_transform(target_train)

scaler_features_finetune = MinMaxScaler(feature_range=(-1, 1))
scaled_features_validation = scaler_features_finetune.fit_transform(features_validation)
scaled_close_validation = scaler_close_train.transform(target_validation)

scaled_features_test = scaler_features_finetune.transform(features_test)
scaled_close_test = scaler_close_train.transform(target_test)

# Función para crear secuencias
def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

SEQ_LENGTH = 60

# Crear secuencias para el entrenamiento inicial
X_train, y_train = create_sequences(scaled_features_train, scaled_close_train, SEQ_LENGTH)
X_validation, y_validation = create_sequences(scaled_features_validation, scaled_close_validation, SEQ_LENGTH)
X_test, y_test = create_sequences(scaled_features_test, scaled_close_test, SEQ_LENGTH)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# 2. Definir el modelo Bi-LSTM
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        att_out = self.attention(out)
        out = self.fc(att_out)
        return out

# Inicializar el modelo
input_size = X_train.shape[2]  # This should now be 7
hidden_size = 64
num_layers = 2
output_size = 1

model = BiLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Entrenamiento Inicial (sin 'Sentiment')
EPOCHS = 20
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(EPOCHS):
    model.train()
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sequences, labels in validation_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            val_loss += criterion(outputs, labels).item()
    
    val_loss /= len(validation_loader)
    scheduler.step(val_loss)
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# 4. Fine-Tuning (con 'Sentiment')
# Ajustar el modelo para aceptar 7 características ahora

# Asegurarse de que las nuevas capas estén en la GPU
model.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                     batch_first=True, bidirectional=True).to(device)
model.fc = nn.Linear(hidden_size * 2, output_size).to(device)

# Reentrenar las nuevas capas
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

EPOCHS_FINE_TUNE = 20
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(EPOCHS_FINE_TUNE):
    model.train()
    for sequences, labels in validation_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sequences, labels in validation_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            val_loss += criterion(outputs, labels).item()
    
    val_loss /= len(validation_loader)
    scheduler.step(val_loss)
    
    print(f'Epoch Fine-Tuning [{epoch+1}/{EPOCHS_FINE_TUNE}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Load the best model for final prediction
model.load_state_dict(torch.load('best_model.pth'))

# Predicción final
input_sequence = scaled_features_test[-SEQ_LENGTH:]  # Use test data instead of validation
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    prediction = model(input_sequence)

# Invertir la escala para 'Close'
predicted_close_price = scaler_close_train.inverse_transform(prediction.cpu().numpy())
actual_close_price = df_test['Close'].iloc[-1]  # Get the last actual close price from test data

print(f"Predicción del precio de cierre: {predicted_close_price[0][0]:.2f}")
print(f"Precio de cierre real: {actual_close_price:.2f}")
print(f"Diferencia: {actual_close_price - predicted_close_price[0][0]:.2f}")

torch.save(model.state_dict(), 'fineTuning_model.pth')

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluation on test set
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Inverse transform the scaled values
predictions_original = scaler_close_train.inverse_transform(all_predictions)
targets_original = scaler_close_train.inverse_transform(all_targets)

# Calculate metrics
mape = mean_absolute_percentage_error(targets_original, predictions_original)
mae = mean_absolute_error(targets_original, predictions_original)
rmse = np.sqrt(mean_squared_error(targets_original, predictions_original))

print(f"Test Set Metrics:")
print(f"MAPE: {mape:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Create a plot
plt.figure(figsize=(12, 6))
plt.plot(targets_original, label='Real Price', color='blue')
plt.plot(predictions_original, label='Predicted Price', color='red')
plt.title('Real vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plt.savefig('real_vs_predicted_prices_BiLSTM.png')
plt.close()

print("Graph saved as 'real_vs_predicted_prices_BiLSTM.png'")
