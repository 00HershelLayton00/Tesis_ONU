"""
FASE 5: Bloques Regionales vs Ideológicos
- Compara coherencia intra-bloque regional vs ideológico
- Heatmap de coherencia para visualizar clusters reales
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from config import (
    CLEAN_CSV_PATH, PLOTS_DIR, BLOQUES, PAISES_ALIANZA
)
from utils import print_progreso, cargar_datos_limpios, coherencia_mcc

print_progreso("FASE 5: Bloques Regionales vs. Ideológicos")

df = cargar_datos_limpios()

# ============================================================
# 5.1 Función para calcular coherencia intra-bloque
# ============================================================
def coherencia_intra_bloque(df, paises_bloque, min_votos=10):
    """
    Calcula MCC promedio entre todos los pares dentro del bloque.
    Solo incluye pares con al menos min_votos compartidos.
    """
    if len(paises_bloque) < 2:
        return np.nan, []
    
    mccs = []
    pares_validos = []
    
    for p1, p2 in combinations(paises_bloque, 2):
        votos_p1 = df[df['stateab'] == p1].set_index('rcid')['vote_code']
        votos_p2 = df[df['stateab'] == p2].set_index('rcid')['vote_code']
        
        common = votos_p1.index.intersection(votos_p2.index)
        if len(common) >= min_votos:
            mcc = coherencia_mcc(votos_p1.loc[common], votos_p2.loc[common])
            if not np.isnan(mcc):
                mccs.append(mcc)
                pares_validos.append((p1, p2))
    
    return np.mean(mccs) if mccs else np.nan, pares_validos

# ============================================================
# 5.2 Calcular cohesión de cada bloque
# ============================================================
print("Calculando cohesión de cada bloque...")
resultados_bloques = {}

for nombre_bloque, paises in BLOQUES.items():
    # Verificar qué países existen realmente en los datos
    paises_presentes = [p for p in paises if p in df['stateab'].cat.categories]
    faltantes = set(paises) - set(paises_presentes)
    
    if faltantes:
        print(f"   ⚠️ {nombre_bloque}: países no encontrados -> {faltantes}")
    
    cohesion, pares = coherencia_intra_bloque(df, paises_presentes)
    resultados_bloques[nombre_bloque] = {
        'cohesion': cohesion,
        'n_paises': len(paises_presentes),
        'n_pares': len(pares),
    }
    print(f"   {nombre_bloque}: MCC={cohesion:.3f} ({len(paises_presentes)} países, {len(pares)} pares)")

# ============================================================
# 5.3 Heatmap de coherencia de un bloque ampliado (Latinoamérica)
# ============================================================
# Definimos países latinoamericanos (códigos ISO3)
LATAM = [
    "ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM",
    "ECU", "SLV", "GTM", "HTI", "HND", "MEX", "NIC", "PAN",
    "PRY", "PER", "URY", "VEN"
]
latam_presentes = [p for p in LATAM if p in df['stateab'].cat.categories]

print(f"\nConstruyendo heatmap de coherencia para {len(latam_presentes)} países latinoamericanos...")

# Matriz de MCC entre todos los países latinoamericanos
mcc_matrix = pd.DataFrame(
    np.eye(len(latam_presentes)),
    index=latam_presentes,
    columns=latam_presentes,
)

for i, p1 in enumerate(latam_presentes):
    votos_p1 = df[df['stateab'] == p1].set_index('rcid')['vote_code']
    
    for j, p2 in enumerate(latam_presentes):
        if i < j:  # Solo triángulo superior
            votos_p2 = df[df['stateab'] == p2].set_index('rcid')['vote_code']
            common = votos_p1.index.intersection(votos_p2.index)
            
            if len(common) >= 10:
                mcc = coherencia_mcc(votos_p1.loc[common], votos_p2.loc[common])
                mcc_matrix.loc[p1, p2] = mcc_matrix.loc[p2, p1] = mcc
            else:
                mcc_matrix.loc[p1, p2] = mcc_matrix.loc[p2, p1] = np.nan

# ============================================================
# 5.4 Visualización 1: Barras comparativas de cohesión
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel izquierdo: Cohesión por bloque
bloques_ordenados = sorted(resultados_bloques.items(), key=lambda x: x[1]['cohesion'] if not np.isnan(x[1]['cohesion']) else -1, reverse=True)
nombres_b = [b[0].replace('_', ' ') for b in bloques_ordenados]
cohesiones = [b[1]['cohesion'] for b in bloques_ordenados]
colores = ['#2196F3' if 'Eje' not in n and 'OTAN' not in n else '#F44336' if 'Eje' in n else '#4CAF50' for n in nombres_b]

bars = axes[0].barh(nombres_b, cohesiones, color=colores)
axes[0].set_xlabel('MCC Promedio Intra-Bloque')
axes[0].set_title('Cohesión de Voto por Bloque', fontweight='bold')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3, axis='x')

# Panel derecho: Heatmap de Latinoamérica
mask = np.triu(np.ones_like(mcc_matrix, dtype=bool), k=1)
sns.heatmap(
    mcc_matrix,
    mask=mask,
    cmap='RdBu_r',
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'MCC'},
    ax=axes[1],
)
axes[1].set_title('Coherencia entre Países Latinoamericanos', fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
bloques_plot_path = os.path.join(PLOTS_DIR, 'bloques_cohesion.png')
plt.savefig(bloques_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico guardado en {bloques_plot_path}")

# ============================================================
# 5.5 Hallazgos
# ============================================================
print("\n📈 HALLAZGOS DE BLOQUES:")
print(f"   Bloque más cohesionado: {nombres_b[0]} (MCC={cohesiones[0]:.3f})")
print(f"   Bloque menos cohesionado: {nombres_b[-1]} (MCC={cohesiones[-1]:.3f})")

# Detectar probable eje ideológico en LATAM (pares con MCC > 0.8)
pares_fuertes = []
for p1, p2 in combinations(latam_presentes, 2):
    val = mcc_matrix.loc[p1, p2]
    if not np.isnan(val) and val > 0.8:
        pares_fuertes.append((p1, p2, val))

if pares_fuertes:
    print(f"\n   Pares latinoamericanos con MCC > 0.8 (probable eje ideológico):")
    for p1, p2, val in sorted(pares_fuertes, key=lambda x: x[2], reverse=True)[:5]:
        print(f"      {p1}-{p2}: {val:.3f}")

print(f"\n✅ FASE 5 COMPLETADA")