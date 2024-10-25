import json
import pandas as pd
from collections import defaultdict

def convertir_sentimiento(sentimiento):
    if sentimiento == 'positive':
        return 1
    elif sentimiento == 'neutral':
        return 0
    elif sentimiento == 'negative':
        return -1
    else:
        return None  


with open('./cosoSentiment.json', 'r') as file:
    noticias = json.load(file)

sentimientos_por_dia = defaultdict(list)

for noticia in noticias:
    fecha = noticia['date']['$date'][:10] 
    sentimiento = convertir_sentimiento(noticia['sentiment'])
    if sentimiento is not None:
        sentimientos_por_dia[fecha].append(sentimiento)

promedio_sentimientos = {fecha: sum(sentimientos) / len(sentimientos) 
                         for fecha, sentimientos in sentimientos_por_dia.items()}

df_sentimientos = pd.DataFrame(list(promedio_sentimientos.items()), columns=['Date', 'Sentiment'])

df_sp500 = pd.read_csv('./SP500_FullSentimentReduced2.csv')

df_combinado = pd.merge(df_sp500, df_sentimientos, on='Date', how='left')

df_combinado.to_csv('SP500_Sentimentv2.csv', index=False)

print("¡Archivo combinado guardado con éxito!")
