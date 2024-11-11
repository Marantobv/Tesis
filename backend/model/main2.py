from flask import Flask, request, jsonify, send_file
import json
import pandas as pd
import torch
import io
import torch.nn as nn
import matplotlib.dates as mdates
from datetime import datetime
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import BERTBaseUncased
import config
from flask_cors import CORS  # Importar CORS
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib
from io import BytesIO
from matplotlib.dates import DateFormatter
import random
import matplotlib.pyplot as plt

app = Flask(__name__)

CORS(app)

SEQ_LENGTH = 60
EPOCHS = 50
EPOCHS_FINE_TUNE = 50
hidden_size = 64
num_layers = 2
output_size = 1

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

def create_sequences(data, target, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)



# MODEL_NAME = "MarantoBv/BERT_Model"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model.to(device)
# model.to(device)
# model.eval()


device = torch.device(config.DEVICE)
model = BERTBaseUncased()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)

def preprocess_text(text):
    if isinstance(text, str):
        return " ".join(text.split())
    else:
        return ""

def classify_text(model, text, tokenizer, device):
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    sentiment = outputs.argmax(axis=1).item()

    if sentiment == 0:
        return "negative"
    elif sentiment == 1:
        return "neutral"
    elif sentiment == 2:
        return "positive"
    
@app.route('/classify_news', methods=['POST'])
def classify_news():
    news_data = request.json

    classified_news = []
    for news in news_data:
        title = preprocess_text(news['title'])
        description = preprocess_text(news['description'])
        combined_text = title + " " + description

        sentiment = classify_text(model, combined_text, tokenizer, device)
        classified_news.append({
            'title': title,
            'description': description,
            'date': news['date'],
            'sentiment': sentiment
        })

    return jsonify(classified_news)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    file.save(filepath)
    return jsonify({"message": "File uploaded successfully!"})

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json
    period_days = data.get('days')
    period_days = int(period_days)
    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    if os.path.exists(filepath):

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        df = pd.read_csv(filepath)
        # Realiza aquí el procesamiento del CSV según sea necesario

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

        test_dates = df_test['Date'].values[SEQ_LENGTH:] 

        X_train, y_train = create_sequences(scaled_features_train, scaled_close_train, SEQ_LENGTH)
        X_validation, y_validation = create_sequences(scaled_features_validation, scaled_close_validation, SEQ_LENGTH)
        X_test, y_test = create_sequences(scaled_features_test, scaled_close_test, SEQ_LENGTH)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        validation_loader = DataLoader(TensorDataset(X_validation, y_validation), batch_size=32, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        input_size_train = X_train.shape[2]
        input_size_finetune = X_validation.shape[2]

        model = BiLSTMModel(input_size=input_size_train, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

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

        last_actual_close = scaler_close_train.inverse_transform([[input_sequence[-1, 1]]])[0][0]

        for i in range(period_days):
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
        next_X_days = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=period_days, freq='B')

        prediction_df = pd.DataFrame({
            'Date': next_X_days,
            'Predicted Close': predictions
        })

        prediction_df['Date'] = pd.to_datetime(prediction_df['Date']).dt.date
        matplotlib.use('Agg')
        plt.figure(figsize=(12, 6))
        plt.plot(historico_15_dias['Date'], historico_15_dias['Close'], 
                label='Historical Prices', color='blue', marker='o')

        plt.plot(prediction_df['Date'], prediction_df['Predicted Close'], 
                label='Predictions', color='red', linestyle='--', marker='o')

        separation_date = historico_15_dias['Date'].iloc[-1]
        plt.axvline(x=separation_date, color='gray', linestyle=':', alpha=0.5)

        plt.title('IndexPrice: Historical Data and Predictions', fontsize=12, pad=15)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Close Price (USD)', fontsize=10)
        plt.grid(True, alpha=0.3)
        #plt.legend()

        plt.gcf().autofmt_xdate()
        date_formatter = DateFormatter("%Y-%m-%d")
        plt.gca().xaxis.set_major_formatter(date_formatter)

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
        
        plt.tight_layout()
        #matplotlib.use('Agg')
        img = BytesIO()
        plt.savefig(img, format='PNG')
        # Por ejemplo, plt.savefig(img, format='PNG')
        img.seek(0)
        return send_file(img, mimetype='image/png')
        #return jsonify({"message": "Data processed successfully!"})
    else:
        return jsonify({"error": "File not found!"}), 404

@app.route('/volatility', methods=['GET'])
def generate_image():
    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Realiza aquí la generación de imagen basada en el CSV

        df['DateFormat'] = pd.to_datetime(df['Date'])

        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        df['Volatility'] = df['Log_Return'].rolling(window=21).std() * np.sqrt(252)

        df['Volatility'].fillna(0, inplace=True)

        plt.figure(figsize=(14, 7))
        plt.plot(df['DateFormat'], df['Volatility'], label='Volatilidad (Ventana de 21 días)', color='blue')
        plt.title('Volatilidad Mensual del Índice')
        plt.xlabel('Fecha')
        plt.ylabel('Volatilidad')
        plt.grid(True)
        img = BytesIO()
        plt.savefig(img, format='PNG')
        # Por ejemplo, plt.savefig(img, format='PNG')
        img.seek(0)
        return send_file(img, mimetype='image/png')
    else:
        return jsonify({"error": "File not found!"}), 404
    
@app.route('/ciclos', methods=['GET'])
def ciclos():
    filepath = os.path.join(UPLOAD_FOLDER, 'data.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Realiza aquí la generación de imagen basada en el CSV
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        df['Signal'] = 0
        df['Signal'][df['SMA_20'] > df['SMA_50']] = 1  # Ciclo alcista
        df['Signal'][df['SMA_20'] < df['SMA_50']] = -1 # Ciclo bajista

        matplotlib.use('Agg')
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df['Close'], label='Precio de Cierre', color='black')
        plt.plot(df['Date'], df['SMA_20'], label='Media Móvil de 20 días', color='blue')
        plt.plot(df['Date'], df['SMA_50'], label='Media Móvil de 50 días', color='red')
        plt.fill_between(df['Date'], df['Close'].min(), df['Close'].max(), 
                        where=(df['Signal'] == 1), color='green', alpha=0.1, label='Ciclo Alcista')
        plt.fill_between(df['Date'], df['Close'].min(), df['Close'].max(), 
                        where=(df['Signal'] == -1), color='red', alpha=0.1, label='Ciclo Bajista')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Muestra cada 6 meses, ajusta según prefieras
        plt.xticks(rotation=45)  # Rota las etiquetas del eje X para que no se superpongan

        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.title('Tendencia de Ciclos Alcistas y Bajistas en el Indice')
        plt.legend()
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='PNG')
        # Por ejemplo, plt.savefig(img, format='PNG')
        img.seek(0)
        return send_file(img, mimetype='image/png')
    else:
        return jsonify({"error": "File not found!"}), 404

if __name__ == '__main__':
    app.run(debug=True)