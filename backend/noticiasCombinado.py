import json
import glob
from datetime import datetime

# Paso 1: Leer todos los archivos JSON
archivos_json = glob.glob("./model/coso/*.json")  # Cambia la ruta al directorio donde están tus archivos JSON

arreglo_combinado = []

for archivo in archivos_json:
    with open(archivo, 'r') as f:
        data = json.load(f)
        
        # Convertir el campo "date" al formato ISODate
        for item in data:
            if "date" in item:
                item["date"] = {"$date": datetime.strptime(item["date"], "%Y-%m-%d").isoformat() + "Z"}
        
        arreglo_combinado.extend(data)

# Paso 3: Guardar el arreglo combinado en un nuevo archivo JSON
with open('combinadoTODO.json', 'w') as f:
    json.dump(arreglo_combinado, f, indent=4)

print("Archivos combinados y fechas convertidas con éxito!")
