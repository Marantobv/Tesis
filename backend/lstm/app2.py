from flask import Flask, jsonify
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from flask_cors import CORS  # Importar CORS

app = Flask(__name__)
CORS(app)

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

# Load the saved model
checkpoint = torch.load('fineTuning.pth')

# Recreate the model architecture
model = BiLSTMModel(
    input_size=checkpoint['input_size_finetune'],
    hidden_size=checkpoint['hidden_size'],
    num_layers=checkpoint['num_layers'],
    output_size=checkpoint['output_size']
)

# Load the model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Load scalers
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
    
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sentiment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain all of these columns: {required_columns}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    last_seq_length_days = df[required_columns[1:]].tail(seq_length).values
    last_day = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    
    return last_seq_length_days, last_day, last_close

@app.route('/predict')
def predict():
    try:
        input_data, last_day, last_close = load_csv_data("pruebaData.csv")
        predicted_price = predict_next_day(input_data)
        
        response = {
            "last_day": last_day.strftime("%Y-%m-%d"),
            "last_close_price": float(last_close),
            "predicted_price": float(predicted_price)
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)