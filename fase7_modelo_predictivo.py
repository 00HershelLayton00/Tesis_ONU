"""
FASE 7: Modelo Predictivo de Votos
- Random Forest + XGBoost para predecir voto de un país
- Usa: historial, voto de referentes, tema, año
- Incluye SHAP para interpretabilidad
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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