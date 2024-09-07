import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import datetime

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

checkpoint = torch.load('fineTuning.pth')

model = BiLSTMModel(
    input_size=checkpoint['input_size_finetune'],
    hidden_size=checkpoint['hidden_size'],
    num_layers=checkpoint['num_layers'],
    output_size=checkpoint['output_size']
)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

scaler_features_finetune = checkpoint['scaler_features_finetune']
scaler_close_train = checkpoint['scaler_close_train']

def predict_next_day(input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    scaled_input = scaler_features_finetune.transform(input_data)
    
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    predicted_close_price = scaler_close_train.inverse_transform(prediction.cpu().numpy())[0][0]
    
    return predicted_close_price

def load_csv_data(file_path, seq_length=60):
    df = pd.read_csv(file_path)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain all of these columns: {required_columns}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    last_seq_length_days = df[required_columns].tail(seq_length).values
    
    return last_seq_length_days, df['Date'].iloc[-1]

if __name__ == "__main__":
    input_data, last_date = load_csv_data("pruebaData.csv")
    
    predicted_price = predict_next_day(input_data)
    
    next_day = last_date + pd.Timedelta(days=1)
    
    print(f"Last date in data: {last_date.date()}")
    print(f"Prediction for: {next_day.date()}")
    print(f"Predicted closing price: ${predicted_price:.2f}")