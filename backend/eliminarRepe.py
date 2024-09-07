import json

# Leer el archivo JSON
with open("./combinadoTODO.json", "r") as file:
    all_news = json.load(file)

# Eliminar duplicados basados en el título
unique_news = []
titles_seen = set()

for news in all_news:
    if news["title"] not in titles_seen:
        unique_news.append(news)
        titles_seen.add(news["title"])

# Guardar las noticias únicas en un nuevo archivo JSON
with open("combiSINREPE.json", "w") as file:
    json.dump(unique_news, file, indent=4)

print("Unique news data saved to unique_news_data.json")