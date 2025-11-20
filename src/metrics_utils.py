# src/metrics_utils.py
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}