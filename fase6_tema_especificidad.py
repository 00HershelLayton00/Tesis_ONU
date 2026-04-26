"""
FASE 6: Tema-Especificidad de la Coherencia
- ¿La coherencia entre países varía según el tema de la resolución?
- Compara Derechos Humanos vs Desarme vs Economía
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (
    CLEAN_CSV_PATH, PLOTS_DIR, TEMAS, PAISES_ALIANZA
)
from utils import print_progreso, cargar_datos_limpios, coherencia_mcc

print_progreso("FASE 6: Tema-Especificidad de la Coherencia")

df = cargar_datos_limpios()

# ============================================================
# 6.1 Definir pares de países a analizar
# ============================================================
PARES = [
    ("USA", "RUS", "USA-Rusia"),
    ("USA", "CHN", "USA-China"),
    ("USA", "GBR", "USA-Reino Unido"),
    ("CHN", "RUS", "China-Rusia"),
    ("USA", "IND", "USA-India"),
    ("USA", "BRA", "USA-Brasil"),
    ("CHN", "BRA", "China-Brasil"),
]

# ============================================================
# 6.2 Calcular MCC por tema para cada par
# ============================================================
resultados_tema = []

for code_a, code_b, label in PARES:
    if code_a not in df['stateab'].cat.categories or code_b not in df['stateab'].cat.categories:
        print(f"   ⚠️ {label}: país no encontrado, saltando")
        continue
    
    for tema_code, tema_nombre in TEMAS.items():
        df_tema = df[df['unres'] == tema_code]
        
        if len(df_tema) < 10:
            continue
        
        votos_a = df_tema[df_tema['stateab'] == code_a].set_index('rcid')['vote_code']
        votos_b = df_tema[df_tema['stateab'] == code_b].set_index('rcid')['vote_code']
        
        common = votos_a.index.intersection(votos_b.index)
        if len(common) >= 5:
            mcc = coherencia_mcc(votos_a.loc[common], votos_b.loc[common])
            resultados_tema.append({
                'par': label,
                'tema': tema_nombre,
                'mcc': mcc,
                'n_votos': len(common),
            })

df_tema = pd.DataFrame(resultados_tema)

# ============================================================
# 6.3 Pivotear para heatmap
# ============================================================
heatmap_data = df_tema.pivot_table(
    index='par', columns='tema', values='mcc', aggfunc='mean'
)

# Reordenar filas
heatmap_data = heatmap_data.reindex([p[2] for p in PARES if p[2] in df_tema['par'].unique()])

# ============================================================
# 6.4 Visualización: Heatmap de temas por par
# ============================================================
fig, ax = plt.subplots(figsize=(12, len(heatmap_data) * 0.8))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'MCC'},
    ax=ax,
)
ax.set_title('Coherencia de Voto por Tema de Resolución', fontsize=14, fontweight='bold')
ax.set_xlabel('Tema')
ax.set_ylabel('Par de Países')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
tema_plot_path = os.path.join(PLOTS_DIR, 'tema_especificidad.png')
plt.savefig(tema_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico guardado en {tema_plot_path}")

# ============================================================
# 6.5 Estadísticas por tema
# ============================================================
print("\n📈 MCC PROMEDIO POR TEMA (todos los pares USA-X):")
for tema in heatmap_data.columns:
    vals = heatmap_data[tema].dropna()
    if len(vals) > 0:
        print(f"   {tema}: {vals.mean():.3f} (±{vals.std():.3f})")

# Tema con mayor dispersión (evidencia de votación "customizada")
print("\n📈 TEMAS CON MAYOR VARIACIÓN ENTRE PARES (más 'políticos'):")
varianza_temas = heatmap_data.var().sort_values(ascending=False)
for tema in varianza_temas.index[:3]:
    print(f"   {tema}: varianza={varianza_temas[tema]:.3f}")

print(f"\n✅ FASE 6 COMPLETADA")