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

api_key_17 = "BCeoo9GF9reW69NU7hmFUznEEMK79egl4wWw44b6"
api_key_18 = "gKaQku8XA0vKDZk8bQAGG4adOWhB7sxHDXGK5WuT"
api_key_19 = "mpj6o54lZ8DZJKJS3t65vIfLfZfl5WAFyPnehPRa"
api_key_20 = "vVKNV51Yl3p0LukGL7dLg5v2db6kTrGF3Xok8vUE"

def get_news_for_date(date, pages):
    news_list = []
    for page in range(1, pages + 1):
        url = f"https://api.marketaux.com/v1/news/all?countries=us&language=en&symbols=TSLA,AMZN,MSFT,AAPL,NVDA,META,GOOGL,JPM,V&filter_entities=true&published_on={date}&page={page}&group_similar=false&api_token={api_key_20}"
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

start_date = datetime.strptime("2024-10-24", "%Y-%m-%d")
total_days = 1
max_requests_per_day = 25

all_news = []

requests_made = 0

current_date = start_date
while requests_made < max_requests_per_day:
    for day in range(total_days):
        date_str = current_date.strftime("%Y-%m-%d")
        pages_to_fetch = 25
        
        news = get_news_for_date(date_str, pages_to_fetch)
        all_news.extend(news)
        
        requests_made += pages_to_fetch
        if requests_made >= max_requests_per_day:
            break
        
        current_date += timedelta(days=1)

with open("2.json", "w") as file:
    json.dump(all_news, file, indent=4)

print("News data saved to news_data.json")

