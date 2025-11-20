# src/config.py
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

DATA_FILE = DATA_DIR / "material.xlsx"

# 随机种子 & 数据集划分
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 默认模型
DEFAULT_MODEL_NAME = "gbrt"   # 可选: rf, gbrt, xgb, svr