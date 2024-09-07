import requests
import json
from datetime import datetime, timedelta

api_key_1 = "X2NDDqQUyXxnJuwh8TWNPTiM2tOBhtnMCMHrF0US"
api_key_2 = "BeLANycdREZBJfG9j3QrgSRO4Tjprx8cUT6YkZm7"
api_key_3 = "ZVnQjDXBfrldBdmUoz8jNiOfXMgHFNKmjvkDknaT"
api_key_4 = "kYF4nB2RfzNXyCZvqML8LqtrQxYKS4UgNizL2ec7"
api_key_5 = "qhI0JZPIKNlYTbJq12Tz3IeVT1OFYgFOsOv5671Q"
api_key_6 = "G8LL4aSyARpQF4CS5SgdqR5Lq8bQiazSVLyjGlrm"
api_key_7 = "YgcLawmwbgodomZhdH03FN5vCrSyK6A9xqX9Q5E1"
api_key_8 = "dCLAWfEck0e0cZ6bmhNeF50zaiTmItxaJHQoYU7z"
api_key_9 = "QYR3NKHUQ5PcSxRmUQXKcbb77ctppHM1rwoeBoeS"
api_key_10 = "dO4McwxcJHOZGBAm7NHVwLvojIQ2sgPw63bmNMu3"
api_key_11 = "5OcrZzYnxsj08ZgPZOSM8JGwyU8DdkBPIko11XUt"
api_key_12 = "FeRxdh2CdUkbghqytHmLCi3pnKZSMoenpbDyPr8M"
api_key_13 = "qkxCjrahUKy4ivNA1vTZM90j8xTRmN2mFpIcwb3v"
api_key_14 = "YSrYCTPq2xJWYHGLsNPK8c64DjC0Tyfsi5EtUU9G"
api_key_15 = "ODobOJgwuh43qCxZMGhyl00rU8o7nvws3MyVz1pP"
api_key_16 = "aKYI2pUA2GE1pnbWPPpLjJVJTuMuh5MVDvfKG5MY"

# Función para obtener noticias de un día específico
def get_news_for_date(date, pages):
    news_list = []
    for page in range(1, pages + 1):
        url = f"https://api.marketaux.com/v1/news/all?countries=us&language=en&symbols=TSLA,AMZN,MSFT,AAPL,NVDA,META,GOOGL,JPM,V&filter_entities=true&published_on={date}&page={page}&group_similar=false&api_token={api_key_10}"
        print(url)
        response = requests.get(url)
        data = response.json()
        for item in data.get("data", []):
            news_list.append({
                "date": date,
                "title": item.get("title"),
                "description": item.get("description")
            })
    return news_list

# Variables de configuración
start_date = datetime.strptime("2024-08-15", "%Y-%m-%d")
total_requests_per_day = 25
total_days = 1
max_requests_per_day = 25

# Lista para almacenar todas las noticias
all_news = []

# Contador para controlar el número de solicitudes
requests_made = 0

# Bucle para obtener noticias
current_date = start_date
while requests_made < max_requests_per_day:
   # print(requests_made)
    for day in range(total_days):
        date_str = current_date.strftime("%Y-%m-%d")
        pages_to_fetch = 25
        
        # Obtener noticias para el día actual
        news = get_news_for_date(date_str, pages_to_fetch)
        all_news.extend(news)
        
        # Incrementar el contador de solicitudes
        requests_made += pages_to_fetch
        if requests_made >= max_requests_per_day:
            break
        
        # Avanzar al siguiente día
        current_date += timedelta(days=1)

# Guardar las noticias en un archivo JSON
with open("10.json", "w") as file:
    json.dump(all_news, file, indent=4)

print("News data saved to news_data.json")

