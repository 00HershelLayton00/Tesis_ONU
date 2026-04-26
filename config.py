"""
config.py
Variables globales del proyecto Tesis ONU.
Modifica solo este archivo para adaptar el proyecto a tu entorno.
"""

# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================
import os

# En Google Colab, monta Drive en /content/drive
# En local, ajusta a tu ruta absoluta
PROJECT_ROOT = "/content/drive/MyDrive/Tesis_ONU"  # Colab
# PROJECT_ROOT = "/home/tu_usuario/Tesis_ONU"       # Linux local
# PROJECT_ROOT = "C:/Users/tu_usuario/Tesis_ONU"    # Windows local

DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR  = os.path.join(PROJECT_ROOT, "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# 2. FUENTE DE DATOS (Harvard Dataverse)
# ============================================================
# DOI: 10.7910/DVN/LEJUQZ  (Voeten, Strezhnev, Bailey)
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/3587027"
RAW_CSV_PATH  = os.path.join(DATA_DIR, "UNVotes_raw.csv")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "UNVotes_clean.csv")
PCA_MATRIX_PATH = os.path.join(DATA_DIR, "pca_matrix.csv")
RESOLUTION_META_PATH = os.path.join(DATA_DIR, "resolution_metadata.csv")

# ============================================================
# 3. PAÍSES CLAVE PARA ANÁLISIS DE ALIANZAS
# ============================================================
# Usa los códigos ISO3 o nombres exactos como aparecen en la columna 'stateab'
PAISES_ALIANZA = {
    "USA":   "Estados Unidos",
    "RUS":   "Rusia (URSS antes de 1992)",
    "CHN":   "China",
    "GBR":   "Reino Unido",
    "FRA":   "Francia",
    "IND":   "India",
    "BRA":   "Brasil",
    "ZAF":   "Sudáfrica",
    "MEX":   "México",
}

# Códigos de la URSS y Rusia (cambia en 1992)
URSS_CODE = "SUN"   # Soviet Union
RUS_CODE  = "RUS"   # Federación Rusa
URSS_RUS_TRANSITION_YEAR = 1992

# ============================================================
# 4. BLOQUES REGIONALES E IDEOLÓGICOS (Pregunta 2)
# ============================================================
BLOQUES = {
    "Mercosur":         ["ARG", "BRA", "PRY", "URY"],
    "OTAN":             ["USA", "GBR", "FRA", "DEU", "CAN", "ITA", "ESP", "TUR"],
    "Eje_Bolivariano":  ["VEN", "CUB", "NIC", "BOL"],
    "BRIICS":           ["BRA", "RUS", "IND", "IDN", "CHN", "ZAF"],
    "Europa_Nordica":   ["SWE", "NOR", "DNK", "FIN", "ISL"],
}

# ============================================================
# 5. TEMAS DE RESOLUCIONES (Columna 'unres')
# ============================================================
# Códigos más relevantes:
TEMAS = {
    "NU":   "Armas Nucleares",
    "DI":   "Desarme General",
    "HR":   "Derechos Humanos",
    "ME":   "Medio Oriente (Israel/Palestina)",
    "EC":   "Desarrollo Económico",
    "CO":   "Colonialismo",
    "SO":   "Descolonización",
}

# ============================================================
# 6. UMBRALES Y PARÁMETROS DE ANÁLISIS
# ============================================================
UMBRAL_COHERENCIA_GRAFO = 0.75   # Para conectar nodos en el grafo
UMBRAL_PARTICIPACION    = 0.50   # Mínima participación para incluir resolución
ENTROPIA_BASE           = 2      # Base del logaritmo (bits)

# ============================================================
# 7. SEMILLA PARA REPRODUCIBILIDAD
# ============================================================
RANDOM_SEED = 42