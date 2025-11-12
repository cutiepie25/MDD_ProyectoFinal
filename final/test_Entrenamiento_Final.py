# language: python
# Configure Test Framework
import os
import math
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Intentar import absoluto (parent dir 'final' no tiene __init__.py -> namespace package allowed)
try:
    from final.Entrenamiento_Final import safe_mape  # absolute import as requested
except Exception:
    # fallback: definir safe_mape localmente (idéntica a la del notebook)
    def safe_mape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = (y_true != 0)
        return np.nan if mask.sum() == 0 else np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Ruta al dataset (asume el test se corre desde final/ o raíz del repo)
CSV_PATHS = [
    os.path.join(os.getcwd(), "data_prepared.csv"),
    os.path.join(os.getcwd(), "final", "data_prepared.csv"),
    os.path.join(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), "data_prepared.csv")
]

def _locate_csv():
    for p in CSV_PATHS:
        if os.path.exists(p):
            return p
    # último intento: relative to this file's directory (use..)
    alt = os.path.join(os.getcwd(), "final", "data_prepared.csv")
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError("No se encontró data_prepared.csv en rutas esperadas. Rutas probadas: {}".format(CSV_PATHS))

CSV = _locate_csv()

@pytest.fixture(scope="module")
def loaded_data():
    df = pd.read_csv(CSV)
    # eliminar columna si existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    assert 'Pay Amt' in df.columns, "El dataset debe contener la columna 'Pay Amt'"
    return df

def test_target_summary_matches_expected(loaded_data):
    Y = loaded_data['Pay Amt']
    desc = Y.describe()
    # comprobar conteo y valores clave del resumen (exacto por los datos suministrados)
    assert int(desc['count']) == 3787, f"count esperado 3787, obtenido {desc['count']}"
    assert int(desc['max']) == 1520925, f"max esperado 1520925, obtenido {desc['max']}"
    # mediana 825 (50%)
    q50 = Y.quantile(0.5)
    assert int(q50) == 825, f"median esperado 825, obtenido {q50}"

def test_train_test_split_sizes(loaded_data):
    X = loaded_data.drop(columns=["Pay Amt"])
    Y = loaded_data["Pay Amt"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    # filas suman total
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    # proporción correcta (aprox)
    expected_test = int(round(0.3 * X.shape[0]))
    assert abs(X_test.shape[0] - expected_test) <= 1

def test_variance_threshold_keeps_expected_number(loaded_data):
    X = loaded_data.drop(columns=["Pay Amt"])
    Y = loaded_data["Pay Amt"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    keep_cols = X_train.columns[vt.get_support()]
    # según salida del notebook se mantienen 119 columnas
    assert len(keep_cols) == 119, f"Se esperaban 119 columnas retenidas, obtenidas: {len(keep_cols)}"
    # confirmar que al menos 1 columna fue eliminada (2 en el notebook)
    drop_cols = [c for c in X_train.columns if c not in keep_cols]
    assert len(drop_cols) >= 1

def test_continuous_and_binary_detection(loaded_data):
    X = loaded_data.drop(columns=["Pay Amt"])
    Y = loaded_data["Pay Amt"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    keep_cols = X_train.columns[vt.get_support()]
    Xtr = X_train[keep_cols].copy()
    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = [c for c in num_cols if set(Xtr[c].unique()) <= {0,1}]
    cont_cols = [c for c in num_cols if c not in bin_cols]
    # verificar que columnas 'Bill Amt' y 'Purch Amt' estén identificadas como continuas
    assert 'Bill Amt' in cont_cols, "'Bill Amt' debería estar en cont_cols"
    assert 'Purch Amt' in cont_cols, "'Purch Amt' debería estar en cont_cols"
    # asegurarse de que el conteo total de num_cols coincide con suma de cont+bin
    assert len(num_cols) == len(cont_cols) + len(bin_cols)

def test_safe_mape_behaviour_and_edgecases():
    # caso con ceros: todos ceros -> devuelve np.nan
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 1.0, 2.0])
    res = safe_mape(y_true, y_pred)
    assert (isinstance(res, float) and np.isnan(res)), "safe_mape debe devolver np.nan cuando todos y_true==0"
    # caso simple: one non-zero
    y_true = np.array([100.0, 0.0])
    y_pred = np.array([50.0, 0.0])
    res = safe_mape(y_true, y_pred)
    assert math.isclose(res, 50.0, rel_tol=1e-8), f"safe_mape simple esperado 50.0, obtenido {res}"

def test_random_forest_baseline_metrics_finite(loaded_data):
    X = loaded_data.drop(columns=["Pay Amt"])
    Y = loaded_data["Pay Amt"]
    # split y eliminar constantes como en notebook
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train)
    keep_cols = X_train.columns[vt.get_support()]
    Xtr = X_train[keep_cols].copy()
    Xte = X_test[keep_cols].copy()
    # usar pocos estimadores para rapidez en tests
    rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(Xtr, Y_train)
    yp = rf.predict(Xte)
    mse = mean_squared_error(Y_test, yp)
    mae = mean_absolute_error(Y_test, yp)
    mape = safe_mape(Y_test.values, yp)
    # comprobar que no hay inf o nan
    assert np.isfinite(mse) and not np.isnan(mse)
    assert np.isfinite(mae) and not np.isnan(mae)
    # mape puede ser nan si todos y_test==0 (no es el caso). Debe ser finito y razonable (muy amplio)
    assert (np.isnan(mape) == False) and np.isfinite(mape), f"MAPE no debe ser inf/NaN, obtenido {mape}"
    # umbral amplio para detectar overflow/valores absurdos
    assert abs(mape) < 1e12, f"MAPE demasiado grande, posible overflow/inverse-transform: {mape}"

pytest.main(['final/test_Entrenamiento_Final.py', '-q', '-rA', '-s'])