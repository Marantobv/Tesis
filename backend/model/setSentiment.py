import pandas as pd
import json
import torch
from model import BERTBaseUncased
import config

device = torch.device(config.DEVICE)
model = BERTBaseUncased()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)
model.eval()

with open('../combinadoTODO.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

def preprocess_text(text):
    if isinstance(text, str):
        return " ".join(text.split())
    else:
        return ""

df['title'] = df['title'].apply(preprocess_text)
df['description'] = df['description'].apply(preprocess_text)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)

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

df['sentiment'] = df.apply(lambda row: classify_text(model, row['title'] + " " + row['description'], tokenizer, device), axis=1)

result = df.to_dict(orient='records')

with open('cosoSentiment.json', 'w') as file:
    json.dump(result, file, indent=4)
