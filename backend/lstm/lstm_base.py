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

# Load the CSV data
df = pd.read_csv('SP500_FullSentimentReduced.csv')

df_train = df[df['Date'] < '2021-01-01']
df_finetune = df[df['Date'] >= '2021-01-01']

validation_split = 0.8 
split_index = int(len(df_finetune) * validation_split)

df_validation = df_finetune.iloc[:split_index]
df_test = df_finetune.iloc[split_index:]

features_train = df_train[['Open', 'Close']].values
target_train = df_train[['Close']].values

features_validation = df_validation[['Open', 'Close', 'Sentiment']].values
target_validation = df_validation[['Close']].values

features_test = df_test[['Open', 'Close', 'Sentiment']].values
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

def create_sequences(data, target, seq_length, pred_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - pred_length + 1):
        seq = data[i:i + seq_length]
        label = target[i + seq_length:i + seq_length + pred_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

SEQ_LENGTH = 60
PRED_LENGTH = 7

test_dates = df_test['Date'].values[SEQ_LENGTH:]

X_train, y_train = create_sequences(scaled_features_train, scaled_close_train, SEQ_LENGTH, PRED_LENGTH)
X_validation, y_validation = create_sequences(scaled_features_validation, scaled_close_validation, SEQ_LENGTH, PRED_LENGTH)
X_test, y_test = create_sequences(scaled_features_test, scaled_close_test, SEQ_LENGTH, PRED_LENGTH)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
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
output_size = PRED_LENGTH

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
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

model.lstm = nn.LSTM(input_size=input_size_finetune, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True).to(device)
model.fc = nn.Linear(hidden_size * 2, output_size).to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS_FINE_TUNE = 50
for epoch in range(EPOCHS_FINE_TUNE):
    model.train()
    for sequences, labels in validation_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch Fine-Tuning [{epoch+1}/{EPOCHS_FINE_TUNE}], Loss: {loss.item():.4f}')

model.eval()
input_sequence = scaled_features_test[-SEQ_LENGTH:]
input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(input_sequence)

predicted_close_prices = scaler_close_train.inverse_transform(prediction.cpu().numpy())
print(f"Predicted close prices for the next 7 days:")
for i, price in enumerate(predicted_close_prices[0]):
    print(f"Day {i+1}: {price:.2f}")