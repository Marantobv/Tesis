from flask import Flask, jsonify, request
import requests
import json
import pandas as pd
import torch
from model import BERTBaseUncased
import config
from transformers import BertTokenizer
from flask_cors import CORS  # Importar CORS

app = Flask(__name__)
CORS(app)

# Configuración del modelo
device = torch.device(config.DEVICE)
model = BERTBaseUncased()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)
model.eval()

# Cargar el tokenizador
tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)

# Función para preprocesar texto
def preprocess_text(text):
    if isinstance(text, str):
        return " ".join(text.split())
    else:
        return ""

# Función para clasificar texto usando el modelo BERT
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

# Ruta para obtener y clasificar noticias
@app.route('/get_and_classify_news', methods=['GET'])
def get_and_classify_news():
    # Parámetros
    date = "2024-10-20"
    api_token = "aKYI2pUA2GE1pnbWPPpLjJVJTuMuh5MVDvfKG5MY"
    base_url = "https://api.marketaux.com/v1/news/all"
    symbols = "TSLA,AMZN,MSFT,AAPL,NVDA,META,GOOGL,JPM,V"
    countries = "us"
    language = "en"
    
    # Almacenar las noticias obtenidas
    news_list = []
    
    # Hacer dos llamadas a la API (page=1 y page=2) para obtener 6 noticias
    for page in range(1, 3):
        url = f"{base_url}?countries={countries}&language={language}&symbols={symbols}&filter_entities=true&published_on={date}&page={page}&group_similar=false&api_token={api_token}"
        print(url)
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json().get('data', [])
            for news in news_data:
                news_item = {
                    "title": preprocess_text(news['title']),
                    "description": preprocess_text(news['description']),
                    "date": date
                }
                news_list.append(news_item)

    # Guardar las noticias en un archivo JSON
    with open('newsFile.json', 'w') as file:
        json.dump(news_list, file, indent=4)

    # Cargar las noticias en un DataFrame
    df = pd.DataFrame(news_list)

    # Clasificar las noticias
    df['sentiment'] = df.apply(lambda row: classify_text(model, row['title'] + " " + row['description'], tokenizer, device), axis=1)

    # Guardar las noticias clasificadas en un nuevo archivo JSON
    result = df.to_dict(orient='records')
    with open('classified_news.json', 'w') as file:
        json.dump(result, file, indent=4)

    return jsonify({"message": "Noticias clasificadas y guardadas correctamente", "classified_news": result})

if __name__ == '__main__':
    app.run(debug=True)
