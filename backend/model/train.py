import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from sklearn.utils import class_weight
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from sklearn.utils import resample


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")

    sentiment_mapping = {
        "positive": 2,
        "neutral": 1,
        "negative": 0
    }
    dfx.sentiment = dfx.sentiment.map(sentiment_mapping)

    # Sobremuestreo de noticias positivas y negativas
    # positive = dfx[dfx.sentiment == 2]
    # neutral = dfx[dfx.sentiment == 1]
    # negative = dfx[dfx.sentiment == 0]
    
    # Asegurarse de que las clases positivas y negativas tengan el mismo número de muestras que las neutras
    # positive_upsampled = resample(positive, replace=True, n_samples=len(neutral), random_state=42)
    # negative_upsampled = resample(negative, replace=True, n_samples=len(neutral), random_state=42)
    
    # dfx = pd.concat([positive_upsampled, neutral, negative_upsampled])

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.2, random_state=42, stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        text=df_train.text.values, target=df_train.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values, target=df_valid.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    # Calcular los pesos de las clases
    class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler, class_weights)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.argmax(outputs, axis=1)
        print(classification_report(targets, outputs, target_names=['negative', 'neutral', 'positive']))
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
