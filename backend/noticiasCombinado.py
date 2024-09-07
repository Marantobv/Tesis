import json
import glob
from datetime import datetime

archivos_json = glob.glob("./model/coso/*.json")

arreglo_combinado = []

for archivo in archivos_json:
    with open(archivo, 'r') as f:
        data = json.load(f)
        
        for item in data:
            if "date" in item:
                item["date"] = {"$date": datetime.strptime(item["date"], "%Y-%m-%d").isoformat() + "Z"}
        
        arreglo_combinado.extend(data)

with open('combinadoTODO.json', 'w') as f:
    json.dump(arreglo_combinado, f, indent=4)

print("Archivos combinados y fechas convertidas con éxito!")
