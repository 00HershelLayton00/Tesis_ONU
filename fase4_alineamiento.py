"""
FASE 4: Análisis de Alineamiento Histórico
- Evolución de la coherencia USA-Rusia-China desde la Guerra Fría
- MCC por año con detección de puntos de inflexión
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (
    CLEAN_CSV_PATH, PLOTS_DIR, RUS_CODE, URSS_RUS_TRANSITION_YEAR,
    PAISES_ALIANZA
)
from utils import (
    print_progreso, cargar_datos_limpios, coherencia_mcc,
    unificar_urss_rusia
)

print_progreso("FASE 4: Evolución del Alineamiento USA-Rusia-China")

df = cargar_datos_limpios()
df = unificar_urss_rusia(df)

# ============================================================
# 4.1 Países a analizar
# ============================================================
PAISES_OBJETIVO = ["USA", "RUS", "CHN"]
nombres = {
    "USA": "Estados Unidos",
    "RUS": "Rusia/URSS", 
    "CHN": "China",
}

# ============================================================
# 4.2 Función para calcular coherencia anual entre dos países
# ============================================================
def coherencia_anual(df, pais_a, pais_b):
    """
    Calcula MCC entre dos países para cada año.
    Retorna DataFrame con columnas: year, mcc, n_votos
    """
    resultados = []
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]
        
        # Obtener votos de cada país en este año
        votos_a = df_year[df_year['stateab'] == pais_a].set_index('rcid')['vote_code']
        votos_b = df_year[df_year['stateab'] == pais_b].set_index('rcid')['vote_code']
        
        # Alinear por rcid
        common_rcids = votos_a.index.intersection(votos_b.index)
        
        if len(common_rcids) < 5:  # Mínimo para ser significativo
            resultados.append({'year': year, 'mcc': np.nan, 'n_votos': len(common_rcids)})
            continue
        
        mcc = coherencia_mcc(votos_a.loc[common_rcids], votos_b.loc[common_rcids])
        resultados.append({'year': year, 'mcc': mcc, 'n_votos': len(common_rcids)})
    
    return pd.DataFrame(resultados)

# ============================================================
# 4.3 Calcular para cada par
# ============================================================
print("Calculando coherencia anual (MCC)...")
par_usa_rus = coherencia_anual(df, "USA", "RUS")
par_usa_chn = coherencia_anual(df, "USA", "CHN")
par_rus_chn = coherencia_anual(df, "RUS", "CHN")

# ============================================================
# 4.4 Detección de puntos de inflexión
# ============================================================
def detectar_puntos_inflexion(serie_mcc, ventana=5, umbral_cambio=0.15):
    """
    Detecta años donde el MCC cambia más de 'umbral_cambio'
    respecto al promedio de la ventana anterior.
    """
    serie_suave = serie_mcc.rolling(window=ventana, center=True, min_periods=3).mean()
    cambios = serie_suave.diff().abs()
    return cambios[cambios > umbral_cambio].index.tolist()

# Eventos históricos para anotar
EVENTOS_HISTORICOS = {
    1991: "Fin de la URSS",
    2001: "9/11",
    2003: "Invasión de Irak",
    2008: "Crisis Financiera",
    2014: "Anexión de Crimea",
    2020: "COVID-19",
    2022: "Invasión de Ucrania",
}

# ============================================================
# 4.5 Visualización
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Panel superior: MCC ---
ax = axes[0]
ax.plot(par_usa_rus['year'], par_usa_rus['mcc'], 
        color='crimson', linewidth=2, marker='o', markersize=3, label=f'USA vs {nombres["RUS"]}')
ax.plot(par_usa_chn['year'], par_usa_chn['mcc'], 
        color='steelblue', linewidth=2, marker='s', markersize=3, label=f'USA vs {nombres["CHN"]}')

# Zona de Guerra Fría
ax.axvspan(1946, 1991, alpha=0.1, color='gray')
ax.text(1968, ax.get_ylim()[1]*0.95, 'Guerra Fría', 
        fontstyle='italic', alpha=0.6, fontsize=10)

# Línea de cero
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

# Eventos históricos
y_min, y_max = ax.get_ylim()
for year, evento in EVENTOS_HISTORICOS.items():
    if year >= df['year'].min():
        ax.axvline(x=year, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.text(year, y_min + 0.02*(y_max-y_min), evento, 
                rotation=90, fontsize=7, verticalalignment='bottom', alpha=0.7)

ax.set_ylabel('MCC (Correlación de Matthews)')
ax.set_title('Evolución histórica del alineamiento en la AGNU', fontsize=14, fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# --- Panel inferior: Número de votos compartidos ---
ax2 = axes[1]
ax2.fill_between(par_usa_rus['year'], par_usa_rus['n_votos'], 
                  alpha=0.3, color='crimson', label='USA-Rusia')
ax2.plot(par_usa_rus['year'], par_usa_rus['n_votos'], 
         color='crimson', linewidth=1)
ax2.fill_between(par_usa_chn['year'], par_usa_chn['n_votos'], 
                  alpha=0.3, color='steelblue', label='USA-China')
ax2.plot(par_usa_chn['year'], par_usa_chn['n_votos'], 
         color='steelblue', linewidth=1)

ax2.set_xlabel('Año')
ax2.set_ylabel('Nº de votos compartidos')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
alineamiento_plot_path = os.path.join(PLOTS_DIR, 'alineamiento_historico.png')
plt.savefig(alineamiento_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico guardado en {alineamiento_plot_path}")

# ============================================================
# 4.6 Resumen de hallazgos
# ============================================================
print("\n📈 RESUMEN DE ALINEAMIENTO:")
print(f"   MCC medio USA-Rusia (Guerra Fría 1946-1991): {par_usa_rus[par_usa_rus['year']<=1991]['mcc'].mean():.3f}")
print(f"   MCC medio USA-Rusia (Post-Guerra Fría 1992+):  {par_usa_rus[par_usa_rus['year']>=1992]['mcc'].mean():.3f}")
print(f"   MCC medio USA-China (Guerra Fría):              {par_usa_chn[par_usa_chn['year']<=1991]['mcc'].mean():.3f}")
print(f"   MCC medio USA-China (Post-Guerra Fría):         {par_usa_chn[par_usa_chn['year']>=1992]['mcc'].mean():.3f}")

# Año de mínimo alineamiento USA-Rusia reciente
recent_usa_rus = par_usa_rus[par_usa_rus['year'] >= 2000]
if not recent_usa_rus['mcc'].isna().all():
    min_year = recent_usa_rus.loc[recent_usa_rus['mcc'].idxmin(), 'year']
    min_val = recent_usa_rus['mcc'].min()
    print(f"\n   Mínimo reciente USA-Rusia: {min_val:.3f} en {int(min_year)}")

print(f"\n✅ FASE 4 COMPLETADA")onstrucción de Matriz PCA")

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