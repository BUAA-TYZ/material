# src/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_FILE, RANDOM_STATE, TEST_SIZE


def _build_material_column(df: pd.DataFrame) -> pd.DataFrame:
    """基于前两列拼接 material 标识。"""
    material_cols = df.columns[:2]
    df = df.copy()
    df["material"] = (
        df[material_cols[0]].astype(str).str.strip()
        + "-"
        + df[material_cols[1]].astype(str).str.strip()
    )
    return df


def load_dataset(path=DATA_FILE):
    """读取训练 Excel，返回 df / X / y / 特征名 / 目标列名"""
    df = pd.read_excel(path)

    # 前两列是材料信息，中间是特征，最后一列是分数
    target_col = df.columns[-1]
    feature_cols = df.columns[2:-1]

    df = _build_material_column(df)
    X = df[feature_cols]
    y = df[target_col]

    return df, X, y, feature_cols, target_col


def load_prediction_dataset(path, feature_cols):
    """读取待预测 Excel，并按训练集特征列顺序返回 X。"""
    df = _build_material_column(pd.read_excel(path))

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"待预测文件缺少训练所需特征列: {missing_cols}"
        )

    X = df[list(feature_cols)]
    return df, X


def train_test_split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """拆分数据集"""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
