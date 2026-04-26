"""
main.py
Script maestro que ejecuta todas las fases en orden.
Útil en Google Colab: simplemente ejecuta !python main.py
"""
import sys
import os
import time

# Asegurar que el directorio del proyecto está en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

fases = [
    ("fase1_obtencion_datos", "Fase 1: Obtención de Datos"),
    ("fase2_limpieza_eda", "Fase 2: Limpieza y EDA"),
    ("fase3_matriz_pca", "Fase 3: Matriz PCA"),
    ("fase4_alineamiento", "Fase 4: Alineamiento Histórico"),
    ("fase5_bloques", "Fase 5: Bloques Regionales vs Ideológicos"),
    ("fase6_tema_especificidad", "Fase 6: Tema-Especificidad"),
    ("fase7_modelo_predictivo", "Fase 7: Modelo Predictivo"),
    ("fase8_redes_puente", "Fase 8: Redes y Países Puente"),
]

if __name__ == "__main__":
    print("=" * 60)
    print("  TESIS ONU - ANÁLISIS COMPLETO DE VOTACIONES")
    print("=" * 60)
    
    tiempo_inicio = time.time()
    
    for modulo, descripcion in fases:
        print(f"\n{'#' * 60}")
        print(f"# EJECUTANDO: {descripcion}")
        print(f"{'#' * 60}")
        
        try:
            exec(f"import {modulo}")
            print(f"✅ {descripcion} completado con éxito")
        except Exception as e:
            print(f"❌ Error en {descripcion}: {e}")
            import traceback
            traceback.print_exc()
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n{'=' * 60}")
    print(f"  PROCESO COMPLETADO en {tiempo_total:.0f} segundos")
    print(f"{'=' * 60}")