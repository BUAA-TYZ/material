# src/models.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# XGBoost 可选依赖
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_model(name: str):
    """
    根据名字返回对应模型实例
    可选: rf, gbrt, xgb, svr
    """
    name = name.lower()

    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "gbrt":
        return GradientBoostingRegressor(
            random_state=42,
        )
    elif name == "svr":
        # 特征没标准化的话注意效果
        return SVR(kernel="rbf", C=10.0, epsilon=0.1)
    elif name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost 未安装，请先 `pip install xgboost`")
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")