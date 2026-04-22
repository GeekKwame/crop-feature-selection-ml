from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.feature_selection import DEFAULT_DATA_PATH, FEATURES, TARGET, evaluate_features, get_best_feature, load_data


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_assets(data_path: str | Path = DEFAULT_DATA_PATH, out_dir: str | Path = "assets") -> dict[str, Path]:
    """
    Generate a few beginner-friendly charts and save them as PNGs.
    Returns a mapping of asset name -> file path.
    """
    df = load_data(data_path)
    out = Path(out_dir)
    _ensure_dir(out)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    assets: dict[str, Path] = {}

    # 1) Crop class distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df[TARGET].value_counts().index
    sns.countplot(data=df, y=TARGET, hue=TARGET, order=order, ax=ax, palette="viridis", legend=False)
    ax.set_title("Crop class distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    fig.tight_layout()
    p = out / "01_crop_class_distribution.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    assets["crop_class_distribution"] = p

    # 2) Feature distributions
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = {"N": "Nitrogen (N)", "P": "Phosphorous (P)", "K": "Potassium (K)", "ph": "pH"}
    palette = sns.color_palette("viridis", 4)
    for ax, feat, color in zip(axes, FEATURES, palette):
        sns.histplot(df[feat], kde=True, ax=ax, color=color, bins=30)
        ax.set_title(labels.get(feat, feat), fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("Distribution of soil features", fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    p = out / "02_feature_distributions.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    assets["feature_distributions"] = p

    # 3) Single-feature model score comparison
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    perf = evaluate_features(X_train, X_test, y_train, y_test, FEATURES)
    best_feature, _ = get_best_feature(perf)

    perf_df = (
        pd.Series(perf, name="Weighted F1")
        .rename_axis("Feature")
        .reset_index()
        .sort_values("Weighted F1", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2a9d8f" if f == best_feature else "#8d99ae" for f in perf_df["Feature"]]
    bars = ax.barh(perf_df["Feature"], perf_df["Weighted F1"], color=colors, edgecolor="white", height=0.55)
    for bar, score in zip(bars, perf_df["Weighted F1"]):
        ax.text(float(score) + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, float(perf_df["Weighted F1"].max()) * 1.25)
    ax.set_xlabel("Weighted F1-score")
    ax.set_title("Which single soil metric predicts crop best?", fontweight="bold", pad=12)
    ax.invert_yaxis()
    fig.tight_layout()
    p = out / "03_single_feature_scores.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    assets["single_feature_scores"] = p

    return assets


def main() -> None:
    assets = save_assets()
    print("[OK] Saved assets:")
    for name, path in assets.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()

