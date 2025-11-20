# src/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_FILE, RANDOM_STATE, TEST_SIZE


def load_dataset(path=DATA_FILE):
    """读取 Excel，构造 material 列，返回 df / X / y / 特征名 / 目标列名"""
    df = pd.read_excel(path)

    # 前两列是材料信息，中间是特征，最后一列是分数
    material_cols = df.columns[:2]
    target_col = df.columns[-1]
    feature_cols = df.columns[2:-1]

    # 拼接材料名
    df["material"] = (
        df[material_cols[0]].astype(str).str.strip()
        + "-"
        + df[material_cols[1]].astype(str).str.strip()
    )

    X = df[feature_cols]
    y = df[target_col]

    return df, X, y, feature_cols, target_col


def train_test_split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """拆分数据集"""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )