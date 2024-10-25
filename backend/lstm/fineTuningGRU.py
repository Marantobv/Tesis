import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv('SP500_FullSentiment.csv')

df_train = df[df['Date'] < '2021-01-01']
df_finetune = df[df['Date'] >= '2021-01-01']

validation_split = 0.8 
split_index = int(len(df_finetune) * validation_split)

df_validation = df_finetune.iloc[:split_index]
df_test = df_finetune.iloc[split_index:]

features_train = df_train[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
target_train = df_train[['Close']].values

features_validation = df_validation[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values
target_validation = df_validation[['Close']].values

features_test = df_test[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values
target_test = df_test[['Close']].values

scaler_features_train = MinMaxScaler(feature_range=(-1, 1))
scaler_close_train = MinMaxScaler(feature_range=(-1, 1))

scaled_features_train = scaler_features_train.fit_transform(features_train)
scaled_close_train = scaler_close_train.fit_transform(target_train)

scaler_features_finetune = MinMaxScaler(feature_range=(-1, 1))
scaled_features_validation = scaler_features_finetune.fit_transform(features_validation)
scaled_close_validation = scaler_close_train.transform(target_validation)

scaled_features_test = scaler_features_finetune.transform(features_test)
scaled_close_test = scaler_close_train.transform(target_test)

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

X_train, y_train = create_sequences(scaled_features_train, scaled_close_train, SEQ_LENGTH)
X_validation, y_validation = create_sequences(scaled_features_validation, scaled_close_validation, SEQ_LENGTH)
X_test, y_test = create_sequences(scaled_features_test, scaled_close_test, SEQ_LENGTH)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

input_size_train = X_train.shape[2]  # 6 características: Open, High, Low, Close, Adj Close, Volume
input_size_finetune = X_validation.shape[2]  # 7 características: Open, High, Low, Close, Adj Close, Volume, Sentiment
hidden_size = 64
num_layers = 2
output_size = 1

model = GRUModel(input_size=input_size_train, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

class GRUModelFineTune(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModelFineTune, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

model = GRUModelFineTune(input_size=input_size_finetune, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS_FINE_TUNE = 50
for epoch in range(EPOCHS_FINE_TUNE):
    model.train()
    epoch_loss = 0
    for sequences, labels in validation_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(validation_loader)
    print(f'Epoch Fine-Tuning [{epoch+1}/{EPOCHS_FINE_TUNE}], Loss: {avg_loss:.4f}')

model.eval()
input_sequence = scaled_features_validation[-SEQ_LENGTH:]
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(input_sequence)

predicted_close_price = scaler_close_train.inverse_transform(prediction.cpu().numpy())
print(f"Predicción del precio de cierre: {predicted_close_price[0][0]}")

torch.save(model.state_dict(), 'fineTuning_model_GRU.pth')

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

predictions_original = scaler_close_train.inverse_transform(all_predictions)
targets_original = scaler_close_train.inverse_transform(all_targets)

mape = mean_absolute_percentage_error(targets_original, predictions_original)
mae = mean_absolute_error(targets_original, predictions_original)
rmse = np.sqrt(mean_squared_error(targets_original, predictions_original))

print(f"Test Set Metrics:")
print(f"MAPE: {mape:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(targets_original, label='Precio Real', color='blue')
plt.plot(predictions_original, label='Precio Predicho', color='red')
plt.title('Precios Reales vs Precios Predichos')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.legend()
plt.grid(True)

plt.savefig('real_vs_predicted_prices_GRU.png')
plt.close()

print("Gráfico guardado como 'real_vs_predicted_prices_GRU.png'")
