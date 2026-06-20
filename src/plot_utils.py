# src/plot_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIG_DIR, TABLE_DIR


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def plot_pred_vs_actual(y_true, y_pred, filename_prefix="pred_vs_actual"):
    ensure_dirs()

    df = pd.DataFrame({"true_score": y_true, "predicted_score": y_pred})
    df.to_csv(TABLE_DIR / f"{filename_prefix}_data.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted vs Actual (Test Set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{filename_prefix}.png", dpi=300)
    plt.close()


def plot_residual_hist(residuals, filename_prefix="residual_distribution"):
    ensure_dirs()

    counts, bins = np.histogram(residuals, bins=20)
    residual_hist_df = pd.DataFrame(
        {
            "bin_left": bins[:-1],
            "bin_right": bins[1:],
            "frequency": counts,
        }
    )
    residual_hist_df.to_csv(
        TABLE_DIR / f"{filename_prefix}_data.csv", index=False
    )

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor="k")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{filename_prefix}.png", dpi=300)
    plt.close()


def plot_feature_importance(importance, feature_names, filename_prefix="feature_importance"):
    ensure_dirs()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(TABLE_DIR / f"{filename_prefix}_data.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{filename_prefix}.png", dpi=300)
    plt.close()

    # 累积重要性
    importance_df["cumulative_importance"] = importance_df["importance"].cumsum()
    importance_df.to_csv(
        TABLE_DIR / "cumulative_importance_data.csv", index=False
    )

    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(importance_df) + 1),
        importance_df["cumulative_importance"],
        marker="o",
    )
    plt.xlabel("Number of Features")
    plt.ylabel("Cumulative Importance")
    plt.title("Cumulative Feature Importance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cumulative_importance.png", dpi=300)
    plt.close()


def plot_top10(df_with_pred, target_col, filename_prefix="top10_predicted_vs_actual"):
    ensure_dirs()

    df_pred_all = df_with_pred.copy()
    top10 = df_pred_all.nlargest(10, "predicted_score")[["material", target_col, "predicted_score"]]
    top10.to_csv(TABLE_DIR / f"{filename_prefix}_data.csv", index=False)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(top10))
    plt.bar(x - 0.2, top10[target_col], width=0.4, label="True", edgecolor="k")
    plt.bar(x + 0.2, top10["predicted_score"], width=0.4, label="Predicted", edgecolor="k")
    plt.xticks(x, top10["material"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Top 10 Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{filename_prefix}.png", dpi=300)
    plt.close()


def plot_cross_validation_metrics(
    fold_results,
    model_name,
    n_splits=10,
    n_repeats=5,
    filename_prefix="cross_validation_metrics",
):
    """绘制重复交叉验证中 R²、RMSE 和 MAE 的逐折分布。"""
    ensure_dirs()
    metric_config = [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")]
    fill_color = "#FFE15B"
    edge_color = "#736422"
    ink_color = "#1F2430"
    muted_color = "#6F768A"
    grid_color = "#E6E8F0"

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6))
    fig.patch.set_facecolor("#FCFCFD")

    for ax, (metric, label) in zip(axes, metric_config):
        ax.set_facecolor("#FFFFFF")
        box = ax.boxplot(
            fold_results[metric],
            patch_artist=True,
            widths=0.45,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": edge_color,
                "markeredgecolor": edge_color,
                "markersize": 4,
            },
            medianprops={"color": ink_color, "linewidth": 1.2},
            whiskerprops={"color": edge_color, "linewidth": 1.0},
            capprops={"color": edge_color, "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#FFFFFF",
                "markeredgecolor": edge_color,
                "markersize": 3,
            },
        )
        box["boxes"][0].set_facecolor(fill_color)
        box["boxes"][0].set_edgecolor(edge_color)
        ax.set_xticks([1], [label])
        ax.set_ylabel(label)
        ax.grid(axis="y", color=grid_color, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D7DBE7")
        ax.spines["bottom"].set_color("#D7DBE7")
        ax.tick_params(colors=muted_color)

    fig.suptitle(
        f"Repeated cross-validation metrics ({model_name.upper()})",
        x=0.08,
        y=0.98,
        ha="left",
        color=ink_color,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.08,
        0.92,
        f"{n_repeats} repeats × {n_splits} folds; each point contributes one validation score",
        ha="left",
        color=muted_color,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.82, wspace=0.42)
    fig.savefig(FIG_DIR / f"{filename_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
