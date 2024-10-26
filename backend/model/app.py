import time
import torch
import flask
from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config
import transformers


app = Flask(__name__)

DEVICE = config.DEVICE
MODEL_NAME = "MarantoBv/BERT_Model"  # Reemplaza con tu nombre de usuario y el nombre del modelo en Hugging Face

# Cargar el modelo y el tokenizador directamente desde Hugging Face
tokenizer = config.TOKENIZER
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
MODEL.to(DEVICE)
MODEL.eval()

def sentence_prediction(sentence):
    # Tokenización y preparación de datos
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    # Mover los tensores al dispositivo configurado
    ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        # Predicción
        outputs = MODEL(input_ids=ids, attention_mask=mask)
        predictions = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    return predictions

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    predictions = sentence_prediction(sentence)
    response = {
        "positive": str(predictions[2]),  # Probabilidad de la clase positiva
        "neutral": str(predictions[1]),   # Probabilidad de la clase neutral
        "negative": str(predictions[0]),  # Probabilidad de la clase negativa
        "sentence": sentence,
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
