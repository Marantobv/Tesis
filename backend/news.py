import requests
import json
from datetime import datetime, timedelta

# Función para obtener noticias y guardarlas en un archivo JSON
def fetch_news(date, api_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={date}T0000&topics=economy_monetary&sort=EARLIEST&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        news_items = []
        
        for news in data.get("feed", []):
            title = news.get("title")
            summary = news.get("summary")
            
            news_items.append({
                "title": title,
                "summary": summary,
                "date": date
            })
        
        # Guardar en archivo JSON
        with open(f'news_{date}.json', 'w') as json_file:
            json.dump(news_items, json_file, indent=4)
    else:
        print(f"Error fetching data for {date}: {response.status_code}")

# Parámetros iniciales
start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
api_key = "2S7L18MPQK16N18Z"
total_days = 25

# Ciclo para realizar 25 solicitudes, una por cada día
for i in range(total_days):
    current_date = (start_date + timedelta(days=i)).strftime("%Y%m%d")
    fetch_news(current_date, api_key)
    print(f"Fetched and saved news for {current_date}")

print("All requests completed and news saved.")
