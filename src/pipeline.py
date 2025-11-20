# src/pipeline.py
import pandas as pd
from pathlib import Path

from .config import OUTPUT_DIR
from .data_utils import load_dataset, train_test_split_data
from .models import get_model
from .metrics_utils import regression_metrics
from .plot_utils import (
    plot_pred_vs_actual,
    plot_residual_hist,
    plot_feature_importance,
    plot_top10,
)


def run_pipeline(model_name: str = "rf"):
    # 1. 读数据
    df, X, y, feature_cols, target_col = load_dataset()

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 3. 获取并训练模型
    model = get_model(model_name)
    model.fit(X_train, y_train)

    # 4. 评估
    y_pred_test = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred_test)
    print(f"[{model_name}] R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")

    # 5. 对全部样本预测 & 保存
    df["predicted_score"] = model.predict(X)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_excel = OUTPUT_DIR / "predicted_with_all_scores.xlsx"
    df.to_excel(out_excel, index=False)
    print(f"✅ 完整预测结果表已保存: {out_excel}")

    # 6. 画图 + 导出原始数据
    # 6.1 预测 vs 实际
    plot_pred_vs_actual(y_test, y_pred_test)

    # 6.2 残差分布
    residuals = y_test - y_pred_test
    plot_residual_hist(residuals)

    # 6.3 特征重要性（只有树模型有 feature_importances_）
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model.feature_importances_, feature_cols)

    # 6.4 Top10 样本预测 vs 实际
    plot_top10(df, target_col)

    return metrics