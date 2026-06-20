# src/pipeline.py
from pathlib import Path

from .config import OUTPUT_DIR, RANDOM_STATE, TABLE_DIR
from .data_utils import load_dataset, load_prediction_dataset, train_test_split_data
from .models import get_model
from .metrics_utils import regression_metrics, repeated_kfold_evaluation
from .plot_utils import (
    plot_cross_validation_metrics,
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

    # 4.1 5 次重复十折交叉验证
    cv_splits = 10
    cv_repeats = 5
    cv_fold_results, cv_summary = repeated_kfold_evaluation(
        get_model(model_name),
        X,
        y,
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=RANDOM_STATE,
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cv_fold_results.to_csv(
        TABLE_DIR / "cross_validation_fold_results.csv", index=False
    )
    cv_summary.to_csv(TABLE_DIR / "cross_validation_metrics.csv", index=False)
    cv_values = cv_summary.set_index("metric")
    print(
        f"[{model_name}] {cv_repeats}×{cv_splits}-fold CV: "
        f"R²={cv_values.loc['r2', 'mean']:.3f}±{cv_values.loc['r2', 'std']:.3f}, "
        f"RMSE={cv_values.loc['rmse', 'mean']:.3f}±{cv_values.loc['rmse', 'std']:.3f}, "
        f"MAE={cv_values.loc['mae', 'mean']:.3f}±{cv_values.loc['mae', 'std']:.3f}"
    )

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

    # 6.5 重复交叉验证指标分布
    plot_cross_validation_metrics(
        cv_fold_results,
        model_name=model_name,
        n_splits=cv_splits,
        n_repeats=cv_repeats,
    )

    return metrics


def predict_external_xlsx(test_path, model_name: str = "rf", output_path=None):
    """
    用训练集全量拟合模型，对外部 Excel 做预测。
    外部 Excel 需要包含与训练集一致的特征列，前两列会用于拼接 material。
    """
    train_df, X_train, y_train, feature_cols, _ = load_dataset()
    model = get_model(model_name)
    model.fit(X_train, y_train)

    test_path = Path(test_path)
    test_df, X_test = load_prediction_dataset(test_path, feature_cols)
    test_df["predicted_score"] = model.predict(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path else OUTPUT_DIR / f"{test_path.stem}_predicted.xlsx"
    test_df.to_excel(output_path, index=False)

    print(f"训练集样本数: {len(train_df)}")
    print(f"待预测样本数: {len(test_df)}")
    print(f"预测结果已保存: {output_path}")

    return output_path
