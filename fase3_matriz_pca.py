"""
FASE 3: Construcción de la Matriz para PCA y Machine Learning
- Filtra resoluciones con suficiente participación
- Crea matriz pivoteada: filas = países, columnas = rcid
- Guarda matriz y metadata de resoluciones
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import (
    CLEAN_CSV_PATH, PCA_MATRIX_PATH, RESOLUTION_META_PATH, 
    UMBRAL_PARTICIPACION
)
from utils import print_progreso, cargar_datos_limpios

print_progreso("FASE 3: Construcción de Matriz PCA")

df = cargar_datos_limpios()
print(f"Datos cargados: {len(df)} registros")

# ============================================================
# 3.1 Filtrar resoluciones con participación suficiente
# ============================================================
# Contar cuántos países votaron (no ausentes) en cada resolución
# 193 es el máximo de países miembros
participacion = df[df['vote_code'] != 8].groupby('rcid').size()
min_votos = UMBRAL_PARTICIPACION * 193  # ~97 países mínimo

rcid_validos = participacion[participacion >= min_votos].index
df_filtrado = df[df['rcid'].isin(rcid_validos)]

print(f"Resoluciones originales: {df['rcid'].nunique()}")
print(f"Resoluciones con ≥{UMBRAL_PARTICIPACION*100:.0f}% participación: {len(rcid_validos)}")

# ============================================================
# 3.2 Crear matriz pivoteada
# ============================================================
# Filas: países (stateab), Columnas: rcid, Valores: vote_code
matrix = df_filtrado.pivot_table(
    index='stateab',
    columns='rcid',
    values='vote_code',
    fill_value=8  # Ausente como valor por defecto (información valiosa)
)

print(f"Matriz PCA generada: {matrix.shape[0]} países × {matrix.shape[1]} resoluciones")

# ============================================================
# 3.3 Guardar matriz y metadata
# ============================================================
matrix.to_csv(PCA_MATRIX_PATH)
print(f"✅ Matriz guardada en {PCA_MATRIX_PATH}")

# Metadata de resoluciones (para saber qué significa cada columna)
columnas_meta = ['rcid', 'year', 'unres', 'short', 'descr']
meta_disponibles = [c for c in columnas_meta if c in df_filtrado.columns]

resolution_meta = (
    df_filtrado[meta_disponibles]
    .drop_duplicates(subset='rcid')
    .set_index('rcid')
    .loc[matrix.columns]  # Asegurar mismo orden que la matriz
)

resolution_meta.to_csv(RESOLUTION_META_PATH)
print(f"✅ Metadata de resoluciones guardada en {RESOLUTION_META_PATH}")

# ============================================================
# 3.4 Vista previa
# ============================================================
print(f"\n📊 Primeras 5 filas × 5 columnas de la matriz:")
print(matrix.iloc[:5, :5])

print(f"\n📊 Distribución de temas en las resoluciones:")
if 'unres' in resolution_meta.columns:
    print(resolution_meta['unres'].value_counts().head(10))

print(f"\n✅ FASE 3 COMPLETADA")