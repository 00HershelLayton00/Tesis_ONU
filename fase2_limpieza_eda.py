"""
FASE 2: Limpieza y Análisis Exploratorio (EDA)
- Limpieza de datos
- Optimización de tipos
- Cálculo de entropía por país
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    RAW_CSV_PATH, CLEAN_CSV_PATH, PLOTS_DIR, PAISES_ALIANZA
)
from utils import print_progreso, entropia_shannon, unificar_urss_rusia

print_progreso("FASE 2: Limpieza y EDA")

# ============================================================
# 2.1 Carga y limpieza
# ============================================================
df = pd.read_csv(RAW_CSV_PATH)
print(f"Registros originales: {len(df)}")

# Eliminar entidades no estatales (Palestina, Santa Sede)
# stateab = 'PSE' es observador, no estado miembro pleno
df = df[df['stateab'] != 'PSE'].copy()

# Mapeo de votos a códigos numéricos
vote_map = {1: 1, 2: 2, 3: 3, 8: 8}
df['vote_code'] = df['vote'].map(vote_map)

# Unificar URSS/Rusia
df = unificar_urss_rusia(df)

# ============================================================
# 2.2 Optimización de tipos (memoria)
# ============================================================
df['vote_code'] = df['vote_code'].astype('int8')
df['year']      = df['year'].astype('int16')
df['rcid']      = df['rcid'].astype('int32')
df['unres']     = df['unres'].astype('category')
df['country']   = df['country'].astype('category')
df['stateab']   = df['stateab'].astype('category')

print(f"Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================
# 2.3 Guardar datos limpios
# ============================================================
df.to_csv(CLEAN_CSV_PATH, index=False)
print(f"✅ Datos limpios guardados en {CLEAN_CSV_PATH}")

# ============================================================
# 2.4 Entropía de Shannon por país
# ============================================================
print("\nCalculando entropía de Shannon por país...")

entropias = df.groupby('stateab', observed=True)['vote_code'].apply(entropia_shannon)
entropias = entropias.sort_values(ascending=False)

# Top 15 países con mayor entropía (más impredecibles)
top_entropia = entropias.head(15)
# Top 15 con menor entropía (más predecibles)
bottom_entropia = entropias.tail(15)

# ============================================================
# 2.5 Visualización
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel izquierdo: Alta entropía
top_entropia.plot(kind='barh', ax=axes[0], color='coral')
axes[0].set_title('Mayor Entropía de Voto\n(Potenciales "Países Puente")', fontsize=12)
axes[0].set_xlabel('Entropía (bits)')
axes[0].invert_yaxis()

# Panel derecho: Baja entropía
bottom_entropia.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Menor Entropía de Voto\n(Voto Altamente Predecible)', fontsize=12)
axes[1].set_xlabel('Entropía (bits)')

plt.tight_layout()
entropia_plot_path = os.path.join(PLOTS_DIR, 'entropia_paises.png')
plt.savefig(entropia_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico guardado en {entropia_plot_path}")

# ============================================================
# 2.6 Estadísticas resumen
# ============================================================
print("\n📈 ESTADÍSTICAS DE ENTROPÍA:")
print(f"   Entropía máxima: {entropias.iloc[0]:.3f} bits ({entropias.index[0]})")
print(f"   Entropía mínima: {entropias.iloc[-1]:.3f} bits ({entropias.index[-1]})")
print(f"   Entropía media:  {entropias.mean():.3f} bits")
print(f"   Mediana:         {entropias.median():.3f} bits")

# Mostrar entropía de países clave
print("\n   Entropía de países clave:")
for code, name in PAISES_ALIANZA.items():
    if code in entropias.index:
        print(f"      {code} ({name}): {entropias[code]:.3f} bits")

print(f"\n✅ FASE 2 COMPLETADA")