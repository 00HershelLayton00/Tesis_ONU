"""
FASE 8: Análisis de Redes y Detección de Países Puente
- Construye grafo donde nodos = países, aristas = coherencia alta
- Calcula Betweenness Centrality para identificar mediadores
- Visualización interactiva del grafo
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from config import (
    CLEAN_CSV_PATH, PLOTS_DIR, UMBRAL_COHERENCIA_GRAFO, PAISES_ALIANZA
)
from utils import print_progreso, cargar_datos_limpios, coherencia_mcc, entropia_shannon

print_progreso("FASE 8: Detección de Países Puente")

df = cargar_datos_limpios()

# ============================================================
# 8.1 Construir matriz de coherencia entre todos los países
# ============================================================
paises = sorted(df['stateab'].cat.categories)
print(f"Construyendo matriz de coherencia para {len(paises)} países...")

# Cargar matriz de coherencia (o calcular si no existe)
mcc_cache_path = os.path.join(os.path.dirname(CLEAN_CSV_PATH), 'mcc_matrix.csv')

if os.path.exists(mcc_cache_path):
    print("📁 Matriz MCC encontrada en caché. Cargando...")
    mcc_full = pd.read_csv(mcc_cache_path, index_col=0)
    # Asegurar que los países existen
    paises_validos = [p for p in paises if p in mcc_full.index]
    mcc_full = mcc_full.loc[paises_validos, paises_validos]
else:
    print("Calculando matriz MCC (esto puede tomar varios minutos)...")
    mcc_full = pd.DataFrame(
        np.eye(len(paises)),
        index=paises,
        columns=paises,
    )
    
    total_pares = len(paises) * (len(paises) - 1) // 2
    contador = 0
    
    for i, p1 in enumerate(paises):
        votos_p1 = df[df['stateab'] == p1].set_index('rcid')['vote_code']
        
        for j, p2 in enumerate(paises):
            if i < j:
                contador += 1
                if contador % 1000 == 0:
                    print(f"   Progreso: {contador}/{total_pares} pares ({100*contador/total_pares:.1f}%)")
                
                votos_p2 = df[df['stateab'] == p2].set_index('rcid')['vote_code']
                common = votos_p1.index.intersection(votos_p2.index)
                
                if len(common) >= 10:
                    mcc = coherencia_mcc(votos_p1.loc[common], votos_p2.loc[common])
                    mcc_full.loc[p1, p2] = mcc_full.loc[p2, p1] = mcc if not np.isnan(mcc) else 0
                else:
                    mcc_full.loc[p1, p2] = mcc_full.loc[p2, p1] = 0
    
    mcc_full.to_csv(mcc_cache_path)
    print(f"✅ Matriz MCC guardada en caché: {mcc_cache_path}")

# ============================================================
# 8.2 Construir grafo
# ============================================================
print(f"\nConstruyendo grafo (umbral MCC > {UMBRAL_COHERENCIA_GRAFO})...")

G = nx.Graph()

# Añadir nodos
for pais in paises:
    entropia = entropia_shannon(df[df['stateab'] == pais]['vote_code'])
    G.add_node(pais, entropia=entropia)

# Añadir aristas donde MCC > umbral
for p1, p2 in combinations(paises, 2):
    mcc = mcc_full.loc[p1, p2]
    if mcc > UMBRAL_COHERENCIA_GRAFO:
        G.add_edge(p1, p2, weight=mcc)

print(f"Grafo construido: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

# ============================================================
# 8.3 Calcular métricas de centralidad
# ============================================================
print("\nCalculando métricas de centralidad...")

# Betweenness Centrality (detecta países puente)
betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

# Eigenvector Centrality (países conectados a países influyentes)
try:
    eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
except nx.PowerIterationFailedConvergence:
    print("⚠️ Eigenvector centrality no convergió. Usando PageRank como alternativa.")
    eigenvector = nx.pagerank(G, weight='weight')

# Degree ponderado (suma de pesos de aristas)
weighted_degree = {n: sum(d['weight'] for _, d in G[n].items()) for n in G.nodes()}

# ============================================================
# 8.4 Top 15 países puente (mayor betweenness)
# ============================================================
betweenness_df = pd.DataFrame({
    'pais': list(betweenness.keys()),
    'betweenness': list(betweenness.values()),
    'eigenvector': [eigenvector.get(p, 0) for p in betweenness.keys()],
    'weighted_degree': [weighted_degree.get(p, 0) for p in betweenness.keys()],
    'entropia': [G.nodes[p]['entropia'] for p in betweenness.keys()],
}).sort_values('betweenness', ascending=False)

# ============================================================
# 8.5 Visualización
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Top Países Puente
top_15 = betweenness_df.head(15)
axes[0, 0].barh(top_15['pais'], top_15['betweenness'], color='teal')
axes[0, 0].set_title('Top 15 Países Puente (Betweenness Centrality)', fontweight='bold')
axes[0, 0].set_xlabel('Centralidad de Intermediación')
axes[0, 0].invert_yaxis()

# Panel 2: Grafo simplificado
# Tomamos top 40 países por betweenness para visualizar
top_paises = betweenness_df.head(40)['pais'].tolist()
G_sub = G.subgraph(top_paises)

pos = nx.spring_layout(G_sub, weight='weight', seed=RANDOM_SEED, k=1.5, iterations=100)
node_sizes = [betweenness[n] * 5000 + 100 for n in G_sub.nodes()]
node_colors = [G_sub.nodes[n]['entropia'] for n in G_sub.nodes()]

nx.draw_networkx_nodes(G_sub, pos, ax=axes[0, 1], 
                       node_size=node_sizes, node_color=node_colors,
                       cmap='viridis', alpha=0.8)
nx.draw_networkx_edges(G_sub, pos, ax=axes[0, 1], 
                       alpha=0.2, edge_color='gray')
nx.draw_networkx_labels(G_sub, pos, ax=axes[0, 1], font_size=7)

axes[0, 1].set_title('Grafo de Coherencia (MCC > 0.75)', fontweight='bold')
axes[0, 1].axis('off')

# Panel 3: Betweenness vs Entropía
axes[1, 0].scatter(
    betweenness_df['entropia'],
    betweenness_df['betweenness'],
    alpha=0.6, c=betweenness_df['weighted_degree'], cmap='plasma',
    edgecolors='black', linewidth=0.5,
)
# Etiquetar países clave
for pais_code in ['USA', 'CHN', 'RUS', 'IND', 'BRA', 'ZAF', 'MEX', 'TUR', 'SAU']:
    if pais_code in betweenness_df['pais'].values:
        row = betweenness_df[betweenness_df['pais'] == pais_code].iloc[0]
        axes[1, 0].annotate(pais_code, (row['entropia'], row['betweenness']),
                           fontsize=9, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points')
axes[1, 0].set_xlabel('Entropía de Voto (bits)')
axes[1, 0].set_ylabel('Betweenness Centrality')
axes[1, 0].set_title('Países Puente: Alta Entropía + Alta Intermediación', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Tabla de top 5
axes[1, 1].axis('off')
tabla_texto = "TOP 5 PAÍSES PUENTE\n\n"
for i, (_, row) in enumerate(top_15.head(5).iterrows(), 1):
    tabla_texto += f"{i}. {row['pais']}\n"
    tabla_texto += f"   Betweenness: {row['betweenness']:.4f}\n"
    tabla_texto += f"   Entropía: {row['entropia']:.2f} bits\n\n"
axes[1, 1].text(0.1, 0.9, tabla_texto, transform=axes[1, 1].transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
redes_plot_path = os.path.join(PLOTS_DIR, 'redes_paises_puente.png')
plt.savefig(redes_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"📊 Gráfico guardado en {redes_plot_path}")

# ============================================================
# 8.6 Guardar métricas
# ============================================================
metricas_path = os.path.join(os.path.dirname(CLEAN_CSV_PATH), 'metricas_red.csv')
betweenness_df.to_csv(metricas_path, index=False)
print(f"✅ Métricas de red guardadas en {metricas_path}")

# ============================================================
# 8.7 Hallazgos
# ============================================================
print("\n📈 PAÍSES PUENTE DETECTADOS (mayor betweenness):")
for i, (_, row) in enumerate(betweenness_df.head(10).iterrows(), 1):
    nombre = PAISES_ALIANZA.get(row['pais'], row['pais'])
    print(f"   {i:2d}. {row['pais']:4s} ({nombre:<20s}): "
          f"Betweenness={row['betweenness']:.4f}, Entropía={row['entropia']:.2f} bits")

print(f"\n📈 CORRELACIONES:")
print(f"   Betweenness vs Entropía:      {betweenness_df['betweenness'].corr(betweenness_df['entropia']):.3f}")
print(f"   Betweenness vs Peso Grado:    {betweenness_df['betweenness'].corr(betweenness_df['weighted_degree']):.3f}")

print(f"\n✅ FASE 8 COMPLETADA")tion_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

from config import (
    CLEAN_CSV_PATH, MODELS_DIR, PLOTS_DIR, RANDOM_SEED, TEMAS, PAISES_ALIANZA
)
from utils import print_progreso, cargar_datos_limpios

print_progreso("FASE 7: Modelo Predictivo de Votos")

df = cargar_datos_limpios()
np.random.seed(RANDOM_SEED)

# ============================================================
# 7.1 Construcción de features
# ============================================================
print("Construyendo matriz de features...")

# Seleccionamos un país objetivo (ajustable)
PAIS_OBJETIVO = "BRA"  # Cambia según el país que quieras predecir
PAISES_REFERENTES = ["USA", "CHN", "RUS"]

# Filtrar resoluciones donde el país objetivo votó
df_pais = df[df['stateab'] == PAIS_OBJETIVO].copy()
df_pais = df_pais.rename(columns={
    'vote_code': 'target',
    'rcid': 'rcid',
})
df_pais = df_pais[['rcid', 'year', 'target', 'unres']].drop_duplicates()

# Merge con votos de países referentes
for ref in PAISES_REFERENTES:
    if ref in df['stateab'].cat.categories:
        votos_ref = df[df['stateab'] == ref][['rcid', 'vote_code']].rename(
            columns={'vote_code': f'voto_{ref}'}
        )
        df_pais = df_pais.merge(votos_ref, on='rcid', how='left')
    else:
        df_pais[f'voto_{ref}'] = np.nan

# Codificar tema como variable numérica
le = LabelEncoder()
df_pais['tema_encoded'] = le.fit_transform(df_pais['unres'].astype(str))

# Feature del año normalizado
df_pais['year_norm'] = (df_pais['year'] - 1946) / (2024 - 1946)

# Eliminar filas con NaN (resoluciones donde algún referente no votó)
df_pais = df_pais.dropna(subset=[f'voto_{r}' for r in PAISES_REFERENTES if f'voto_{r}' in df_pais.columns])
print(f"Registros para modelo: {len(df_pais)}")

# ============================================================
# 7.2 Preparar X e y
# ============================================================
feature_cols = [f'voto_{r}' for r in PAISES_REFERENTES if f'voto_{r}' in df_pais.columns]
feature_cols += ['tema_encoded', 'year_norm']

X = df_pais[feature_cols].copy()
y = df_pais['target'].copy()

# Mapear votos a clases contiguas (0,1,2,3) para clasificación
# 1->0, 2->1, 3->2, 8->3
voto_a_clase = {1: 0, 2: 1, 3: 2, 8: 3}
clase_a_voto = {v: k for k, v in voto_a_clase.items()}
y = y.map(voto_a_clase)

# Train-test split temporal (respetando orden cronológico)
split_year = int(np.percentile(df_pais['year'], 80))  # 80% para train, 20% para test
X_train = X[df_pais['year'] <= split_year]
y_train = y[df_pais['year'] <= split_year]
X_test  = X[df_pais['year'] > split_year]
y_test  = y[df_pais['year'] > split_year]

print(f"Train: {len(X_train)} muestras (hasta {split_year})")
print(f"Test:  {len(X_test)} muestras (desde {split_year+1})")
print(f"Distribución de clases en train:")
print(y_train.value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

# ============================================================
# 7.3 Random Forest
# ============================================================
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.3f}")
print("\nClassification Report:")
clases_nombres = ['Sí', 'No', 'Abstención', 'Ausente']
print(classification_report(
    y_test, y_pred_rf,
    target_names=[clases_nombres[i] for i in range(4) if i in y_test.unique()],
    zero_division=0,
))

# Guardar modelo
modelo_rf_path = os.path.join(MODELS_DIR, f'random_forest_{PAIS_OBJETIVO}.pkl')
with open(modelo_rf_path, 'wb') as f:
    pickle.dump({'modelo': rf, 'label_encoder': le, 'feature_cols': feature_cols}, f)
print(f"✅ Modelo guardado en {modelo_rf_path}")

# ============================================================
# 7.4 XGBoost
# ============================================================
print("\n--- XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric='mlogloss',
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {acc_xgb:.3f}")

# Guardar modelo
modelo_xgb_path = os.path.join(MODELS_DIR, f'xgboost_{PAIS_OBJETIVO}.pkl')
with open(modelo_xgb_path, 'wb') as f:
    pickle.dump({'modelo': xgb_model, 'label_encoder': le, 'feature_cols': feature_cols}, f)
print(f"✅ Modelo guardado en {modelo_xgb_path}")

# ============================================================
# 7.5 Validación cruzada
# ============================================================
print("\n--- Validación Cruzada (5-fold) ---")
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Random Forest CV Accuracy: {cv_scores_rf.mean():.3f} (±{cv_scores_rf.std():.3f})")
print(f"XGBoost CV Accuracy:       {cv_scores_xgb.mean():.3f} (±{cv_scores_xgb.std():.3f})")

# ============================================================
# 7.6 SHAP para interpretabilidad
# ============================================================
print("\nCalculando valores SHAP...")
try:
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Si shap_values es lista (multiclase), tomamos la clase más frecuente
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[0]
    else:
        shap_values_plot = shap_values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values_plot, X_test,
        feature_names=feature_cols,
        class_names=clases_nombres,
        show=False,
    )
    
    shap_plot_path = os.path.join(PLOTS_DIR, f'shap_importancia_{PAIS_OBJETIVO}.png')
    plt.savefig(shap_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 SHAP guardado en {shap_plot_path}")
    
    # Importancia de features
    importancia = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    print("\n📈 IMPORTANCIA DE FEATURES (Random Forest):")
    for _, row in importancia.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
        
except Exception as e:
    print(f"⚠️ Error con SHAP: {e}")
    print("   Continuando sin gráfico SHAP...")

# ============================================================
# 7.7 Resumen
# ============================================================
print(f"\n📈 RESUMEN DE MODELOS PARA {PAIS_OBJETIVO}:")
print(f"   Random Forest: Accuracy = {acc_rf:.3f}, CV = {cv_scores_rf.mean():.3f}±{cv_scores_rf.std():.3f}")
print(f"   XGBoost:       Accuracy = {acc_xgb:.3f}, CV = {cv_scores_xgb.mean():.3f}±{cv_scores_xgb.std():.3f}")
print(f"\n✅ FASE 7 COMPLETADA")