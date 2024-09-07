# FINE TUNING WITH BIDIRECTIONAL LSTM
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random

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

test_dates = df_test['Date'].values[SEQ_LENGTH:] 

X_train, y_train = create_sequences(scaled_features_train, scaled_close_train, SEQ_LENGTH)
X_validation, y_validation = create_sequences(scaled_features_validation, scaled_close_validation, SEQ_LENGTH)
X_test, y_test = create_sequences(scaled_features_test, scaled_close_test, SEQ_LENGTH)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 para bidireccional
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out

input_size_train = X_train.shape[2]
input_size_finetune = X_validation.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1

model = BiLSTMModel(input_size=input_size_train, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

EPOCHS = 50
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

model.lstm = nn.LSTM(input_size=input_size_finetune, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True).to(device)
model.fc = nn.Linear(hidden_size * 2, output_size).to(device)

# Reentrenar las nuevas capas
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS_FINE_TUNE = 50
for epoch in range(EPOCHS_FINE_TUNE):
    model.train()
    for sequences, labels in validation_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch Fine-Tuning [{epoch+1}/{EPOCHS_FINE_TUNE}], Loss: {loss.item():.4f}')

model.eval()
input_sequence = scaled_features_test[-SEQ_LENGTH:].copy()
predictions = []

period = 7

for _ in range(period):
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_scaled = prediction.cpu().numpy()
    
    predicted_close_price = scaler_close_train.inverse_transform(predicted_scaled)[0][0]
    predictions.append(predicted_close_price)
    
    new_row = input_sequence[-1, :].copy()
    new_row[3] = predicted_scaled[0][0] 
    new_row[4] = predicted_scaled[0][0] 
    
    input_sequence = np.vstack((input_sequence[1:], new_row))

last_known_date = pd.to_datetime(test_dates[-1])

next_X_days = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=period, freq='B')

prediction_df = pd.DataFrame({
    'Date': next_X_days,
    'Predicted Close': predictions
})

print("Predictions:")
print(prediction_df)

real_values = np.array([5475.089844, 5509.009766, 5537.02002, 5567.189941, 5572.850098, 5633.910156, 5584.540039])

mape = np.mean(np.abs((real_values - predictions) / real_values)) * 100

mae = mean_absolute_error(real_values, predictions)

rmse = np.sqrt(mean_squared_error(real_values, predictions))

print("\nError Metrics:")
print(f"MAPE: {mape:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

comparison_df = pd.DataFrame({
    'Date': next_X_days,
    'Predicted Close': predictions,
    'Real Close': real_values
})

print("\nComparison of Predicted vs Real Values:")
print(comparison_df)