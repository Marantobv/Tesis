from flask import Flask, jsonify
from flask_cors import CORS  # Importar CORS
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random

app = Flask(__name__)
CORS(app)

def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

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

@app.route('/predict')
def predict():

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

    features_train = df_train[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
    target_train = df_train[['Close']].values

    features_finetune = df_finetune[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']].values
    target_finetune = df_finetune[['Close']].values

    scaler_features_train = MinMaxScaler(feature_range=(-1, 1))
    scaler_close_train = MinMaxScaler(feature_range=(-1, 1))

    scaled_features_train = scaler_features_train.fit_transform(features_train)
    scaled_close_train = scaler_close_train.fit_transform(target_train)

    scaler_features_finetune = MinMaxScaler(feature_range=(-1, 1))
    scaled_features_finetune = scaler_features_finetune.fit_transform(features_finetune)
    scaled_close_finetune = scaler_close_train.transform(target_finetune)

    SEQ_LENGTH = 60

    X_train, y_train = create_sequences(scaled_features_train, scaled_close_train, SEQ_LENGTH)
    X_finetune, y_finetune = create_sequences(scaled_features_finetune, scaled_close_finetune, SEQ_LENGTH)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    finetune_loader = DataLoader(TensorDataset(X_finetune, y_finetune), batch_size=32, shuffle=False)

    input_size_train = X_train.shape[2]
    input_size_finetune = X_finetune.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = BiLSTMModel(input_size=input_size_train, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    EPOCHS = 20

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
    model.fc = nn.Linear(hidden_size * 2, output_size).to(device)  # *2 because of bidirectional

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS_FINE_TUNE = 20
    for epoch in range(EPOCHS_FINE_TUNE):
        model.train()
        for sequences, labels in finetune_loader:
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
    input_sequence = scaled_features_finetune[-SEQ_LENGTH:]
    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_sequence)

    predicted_close_price = scaler_close_train.inverse_transform(prediction.cpu().numpy())
    
    last_close_value = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    
    print(f"Predicción del precio de cierre: {predicted_close_price[0][0]}")
    print(f"Último valor de cierre: {last_close_value}")
    
    return jsonify({
        'predicted_close_price': float(predicted_close_price[0][0]),
        'last_close_value': float(last_close_value),
        'last_date': last_date
    })

if __name__ == '__main__':
    app.run(debug=True)
