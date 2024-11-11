# FINE TUNING WITH BIDIRECTIONAL LSTM
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv('./SP500_FullSentimentReduced.csv')

df_train = df[df['Date'] < '2021-01-01']
df_finetune = df[df['Date'] >= '2021-01-01']
historico_15_dias = df[['Date', 'Close']].iloc[-15:].copy()
historico_15_dias['Date'] = pd.to_datetime(historico_15_dias['Date']).dt.date

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
input_sequence = scaled_features_test[-SEQ_LENGTH:].copy()
predictions = []
raw_predictions = []


period = 7
last_actual_close = scaler_close_train.inverse_transform([[input_sequence[-1, 1]]])[0][0]

for i in range(period):
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_scaled = prediction.cpu().numpy()
    
    raw_predictions.append(predicted_scaled[0][0])
    
    predicted_close_price = scaler_close_train.inverse_transform([[predicted_scaled[0][0]]])[0][0]
    predictions.append(predicted_close_price)
    
    new_row = np.zeros_like(input_sequence[-1, :])
    
    if i == 0:
        previous_close = last_actual_close
    else:
        previous_close = predictions[-1]
    
    open_price = previous_close * (1 + np.random.uniform(-0.005, 0.005))
    
    scaled_open = scaler_features_train.transform([[open_price, open_price]])[0][0]
    
    new_row[0] = scaled_open  # Open
    new_row[1] = predicted_scaled[0][0]  # Close
    
    if input_sequence.shape[1] > 2:
        new_row[2] = np.mean(input_sequence[-5:, 2]) + np.random.uniform(-0.1, 0.1)
    
    input_sequence = np.vstack((input_sequence[1:], new_row))

last_known_date = pd.to_datetime(test_dates[-1])
next_X_days = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=period, freq='B')

prediction_df = pd.DataFrame({
    'Date': next_X_days,
    'Predicted Close': predictions,
    #'Day-over-Day Change': [0] + [((predictions[i] - predictions[i-1])/predictions[i-1])*100 for i in range(1, len(predictions))]
})

print(prediction_df)

prediction_df['Date'] = pd.to_datetime(prediction_df['Date']).dt.date


plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(historico_15_dias['Date'], historico_15_dias['Close'], 
         label='Historical Prices', color='blue', marker='o')

# Plot predictions
plt.plot(prediction_df['Date'], prediction_df['Predicted Close'], 
         label='Predictions', color='red', linestyle='--', marker='o')

# Add vertical line to separate historical data from predictions
separation_date = historico_15_dias['Date'].iloc[-1]
plt.axvline(x=separation_date, color='gray', linestyle=':', alpha=0.5)

# Customize the plot
plt.title('S&P 500 Price: Historical Data and Predictions', fontsize=12, pad=15)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Close Price (USD)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()

# Format x-axis dates
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
date_formatter = DateFormatter("%Y-%m-%d")
plt.gca().xaxis.set_major_formatter(date_formatter)

# Add price annotations
for i, row in historico_15_dias.iterrows():
    plt.annotate(f'${row["Close"]:.2f}', 
                (row['Date'], row['Close']),
                textcoords="offset points",
                xytext=(0,10),
                ha='center',
                fontsize=8)

for i, row in prediction_df.iterrows():
    plt.annotate(f'${row["Predicted Close"]:.2f}', 
                (row['Date'], row['Predicted Close']),
                textcoords="offset points",
                xytext=(0,-15),
                ha='center',
                fontsize=8)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print the numerical data
print("\nHistorical Data (Last 15 days):")
print(historico_15_dias)
print("\nPredicted Values (Next 7 days):")
print(prediction_df)

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

# std_dev = np.std(predictions)
# prediction_df['Lower Bound'] = prediction_df['Predicted Close'] - (1.96 * std_dev)
# prediction_df['Upper Bound'] = prediction_df['Predicted Close'] + (1.96 * std_dev)

# print("\nDetailed Predictions:")
# print(prediction_df.to_string(float_format=lambda x: '{:.2f}'.format(x)))



# prediction_df.rename(columns={'Predicted Close': 'Close'}, inplace=True)
# combinado = pd.concat([historico_15_dias, prediction_df])

# print(combinado)

# plt.figure(figsize=(12, 6))
# plt.plot(combinado['Date'], combinado['Close'], label='Datos Históricos + Predicciones', color='purple', marker='o')
# plt.axvline(x=historico_15_dias['Date'].iloc[-1], color='gray', linestyle='--', label='Inicio de Predicción')

# # Formato de fechas en el eje X
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
# plt.xticks(rotation=45)

# # Etiquetas y leyenda
# plt.xlabel('Fecha')
# plt.ylabel('Precio de Cierre')
# plt.title('Últimos 15 Días + 7 Días de Predicción')
# plt.legend()
# plt.tight_layout()
# plt.show()
