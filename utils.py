"""
utils.py
Funciones de utilidad compartidas por todos los scripts de fase.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from scipy.stats import entropy
from typing import Tuple, List
import config

def coherencia_simple(votos_a: pd.Series, votos_b: pd.Series) -> float:
    """
    Índice de Coincidencia (Agreement Index) simple: 
    (votos iguales) / (total de votos válidos comparables)
    
    Args:
        votos_a, votos_b: Series de pandas alineadas por índice (rcid)
                          Valores: 1=Sí, 2=No, 3=Abstención, 8=Ausente
    Returns:
        float entre 0 y 1
    """
    # Filtramos solo donde AMBOS tienen voto válido (excluimos si ambos ausentes)
    mascara_valida = (votos_a != 8) | (votos_b != 8)
    a = votos_a[mascara_valida]
    b = votos_b[mascara_valida]
    
    if len(a) == 0:
        return np.nan
    
    iguales = (a == b).sum()
    return iguales / len(a)


def coherencia_mcc(votos_a: pd.Series, votos_b: pd.Series) -> float:
    """
    Coeficiente de Correlación de Matthews (MCC) entre dos países.
    Trata la ausencia (8) como una categoría válida.
    
    Returns:
        float entre -1 (oposición perfecta) y +1 (alineamiento perfecto)
    """
    mascara_valida = (votos_a != 8) | (votos_b != 8)
    a = votos_a[mascara_valida]
    b = votos_b[mascara_valida]
    
    if len(a) < 10:  # Mínimo para tener sentido estadístico
        return np.nan
    
    return matthews_corrcoef(a, b)


def entropia_shannon(votos: pd.Series, base: int = None) -> float:
    """
    Calcula la entropía de Shannon de la distribución de votos de un país.
    
    Args:
        votos: Serie con valores de voto (1,2,3,8)
        base: Base del logaritmo (default: config.ENTROPIA_BASE)
    Returns:
        float (bits si base=2, nats si base=e)
    """
    if base is None:
        base = config.ENTROPIA_BASE
    
    counts = votos.value_counts(normalize=True)
    # Aseguramos los 4 estados posibles
    probs = np.array([
        counts.get(1, 0),
        counts.get(2, 0),
        counts.get(3, 0),
        counts.get(8, 0),
    ])
    return entropy(probs, base=base)


def unificar_urss_rusia(df: pd.DataFrame, 
                         code_col: str = "stateab",
                         year_col: str = "year") -> pd.DataFrame:
    """
    Reemplaza el código de URSS por RUS en años <= 1991 para
    tener una serie histórica unificada de Rusia.
    """
    df = df.copy()
    mascara_urss = (df[code_col] == config.URSS_CODE) & (df[year_col] < config.URSS_RUS_TRANSITION_YEAR)
    df.loc[mascara_urss, code_col] = config.RUS_CODE
    return df


def cargar_datos_limpios() -> pd.DataFrame:
    """
    Carga el CSV limpio y aplica tipos optimizados.
    """
    df = pd.read_csv(config.CLEAN_CSV_PATH)
    
    # Restaurar tipos categóricos
    for col in ['country', 'stateab', 'unres']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    if 'vote_code' in df.columns:
        df['vote_code'] = df['vote_code'].astype('int8')
    
    if 'year' in df.columns:
        df['year'] = df['year'].astype('int16')
    
    return df


def print_progreso(mensaje: str):
    """Imprime mensaje de progreso formateado."""
    print(f"\n{'='*60}")
    print(f"  {mensaje}")
    print(f"{'='*60}")