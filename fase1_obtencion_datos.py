"""
FASE 1: Obtención de Datos
Descarga el dataset desde Harvard Dataverse a Google Drive.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import pandas as pd
from config import DATAVERSE_URL, RAW_CSV_PATH, DATA_DIR
from utils import print_progreso

print_progreso("FASE 1: Descargando datos de Harvard Dataverse")

if os.path.exists(RAW_CSV_PATH):
    print(f"📁 {RAW_CSV_PATH} ya existe. Saltando descarga.")
    df = pd.read_csv(RAW_CSV_PATH)
else:
    print(f"⏳ Descargando desde {DATAVERSE_URL} ...")
    response = requests.get(DATAVERSE_URL, stream=True, timeout=120)
    
    if response.status_code != 200:
        raise ConnectionError(f"Error HTTP {response.status_code}. Verifica la URL o tu conexión.")
    
    # Leer directamente el CSV desde la respuesta HTTP
    from io import StringIO
    df = pd.read_csv(StringIO(response.text), low_memory=False)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"✅ Datos descargados ({len(df)} registros)")

print(f"📊 Dimensiones: {df.shape}")
print(f"📋 Columnas: {df.columns.tolist()}")
print(f"📅 Rango de años: {df['year'].min()} - {df['year'].max()}")
print(f"🌍 Países únicos: {df['country'].nunique()}")
print(f"\n✅ FASE 1 COMPLETADA - Datos guardados en {RAW_CSV_PATH}")