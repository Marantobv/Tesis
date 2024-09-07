import json
import pandas as pd
from collections import defaultdict

# Función para convertir el sentimiento en un valor numérico
def convertir_sentimiento(sentimiento):
    if sentimiento == 'positive':
        return 1
    elif sentimiento == 'neutral':
        return 0
    elif sentimiento == 'negative':
        return -1
    else:
        return None  # En caso de otro tipo de sentimiento

# Cargar datos del JSON de noticias
with open('./cosoSentiment.json', 'r') as file:
    noticias = json.load(file)

# Crear diccionario para almacenar los sentimientos por fecha
sentimientos_por_dia = defaultdict(list)

# Procesar los datos del JSON
for noticia in noticias:
    fecha = noticia['date']['$date'][:10]  # Extraer la fecha en formato YYYY-MM-DD
    sentimiento = convertir_sentimiento(noticia['sentiment'])
    if sentimiento is not None:
        sentimientos_por_dia[fecha].append(sentimiento)

# Calcular el promedio del sentimiento por día
promedio_sentimientos = {fecha: sum(sentimientos) / len(sentimientos) 
                         for fecha, sentimientos in sentimientos_por_dia.items()}

# Convertir a DataFrame para facilitar la unión con el CSV del SP500
df_sentimientos = pd.DataFrame(list(promedio_sentimientos.items()), columns=['Date', 'Sentiment'])

# Cargar el archivo CSV del SP500
df_sp500 = pd.read_csv('../lstm/SP500_Sentimentv2.csv')

# Unir ambos DataFrames usando la columna 'Date'
df_combinado = pd.merge(df_sp500, df_sentimientos, on='Date', how='left')

# Guardar el resultado en un nuevo archivo CSV
df_combinado.to_csv('SP500_Sentimentv2.csv', index=False)

print("¡Archivo combinado guardado con éxito!")
