# src/metrics_utils.py
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold


def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def repeated_kfold_evaluation(
    model,
    X,
    y,
    n_splits=10,
    n_repeats=5,
    random_state=42,
):
    """执行重复 K 折交叉验证，返回逐折结果和汇总统计。"""
    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    fold_results = []

    for split_index, (train_index, validation_index) in enumerate(cv.split(X)):
        fitted_model = clone(model)
        fitted_model.fit(X.iloc[train_index], y.iloc[train_index])
        predictions = fitted_model.predict(X.iloc[validation_index])
        metrics = regression_metrics(y.iloc[validation_index], predictions)
        fold_results.append(
            {
                "repeat": split_index // n_splits + 1,
                "fold": split_index % n_splits + 1,
                "train_size": len(train_index),
                "validation_size": len(validation_index),
                **metrics,
            }
        )

    fold_df = pd.DataFrame(fold_results)
    metric_names = ("r2", "rmse", "mae")
    summary_df = pd.DataFrame(
        {
            "metric": metric_names,
            "mean": [fold_df[col].mean() for col in metric_names],
            "std": [fold_df[col].std(ddof=1) for col in metric_names],
            "min": [fold_df[col].min() for col in metric_names],
            "max": [fold_df[col].max() for col in metric_names],
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "n_evaluations": len(fold_df),
        }
    )
    return fold_df, summary_df
