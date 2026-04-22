"""
feature_selection.py
====================
Standalone module for evaluating individual soil features (N, P, K, ph) as
predictors of optimal crop type using Logistic Regression.

Usage (CLI):
    python src/feature_selection.py
    python src/feature_selection.py --data data/soil_measures.csv --features N P K ph

Usage (library):
    from src.feature_selection import load_data, evaluate_features, get_best_feature
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "soil_measures.csv"
FEATURES = ["N", "P", "K", "ph"]
TARGET = "crop"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the soil measures CSV.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with expected columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing from the dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_cols = set(FEATURES + [TARGET])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def evaluate_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    features: list[str] | None = None,
) -> dict[str, float]:
    """Train and evaluate a Logistic Regression model for each individual feature.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices for training and testing.
    y_train, y_test : pd.Series
        Target labels for training and testing.
    features : list of str, optional
        Subset of feature columns to evaluate. Defaults to FEATURES.

    Returns
    -------
    dict[str, float]
        Mapping of feature name -> weighted F1-score.
    """
    if features is None:
        features = FEATURES

    performance: dict[str, float] = {}

    for feature in features:
        # Scale is important for stable convergence with LBFGS.
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        model.fit(X_train[[feature]], y_train)
        y_pred = model.predict(X_test[[feature]])
        f1 = metrics.f1_score(y_test, y_pred, average="weighted")
        performance[feature] = round(f1, 4)

    return performance


def get_best_feature(performance: dict[str, float]) -> tuple[str, float]:
    """Return the feature with the highest F1-score.

    Parameters
    ----------
    performance : dict[str, float]
        Output of :func:`evaluate_features`.

    Returns
    -------
    tuple[str, float]
        (feature_name, f1_score) of the best performing feature.
    """
    best = max(performance, key=performance.get)
    return best, performance[best]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify the most predictive soil feature for crop selection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the soil_measures CSV (default: data/soil_measures.csv)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=FEATURES,
        help=f"Soil features to evaluate (default: {FEATURES})",
    )
    return parser.parse_args(argv)


def main(argv=None) -> dict[str, float]:
    """Run the full feature selection pipeline and print results."""
    args = _parse_args(argv)

    print(f"\nLoading dataset from: {args.data}")
    df = load_data(args.data)
    print(f"  Rows: {len(df):,}  |  Crops: {df[TARGET].nunique()}  |  Features: {args.features}\n")

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    performance = evaluate_features(X_train, X_test, y_train, y_test, args.features)

    print("Feature F1-scores (weighted):")
    print("-" * 30)
    for feat, score in sorted(performance.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 30)
        print(f"  {feat:>4}  {score:.4f}  |{bar}")

    best_feature, best_score = get_best_feature(performance)
    print("-" * 30)
    print(f"\n  Best predictive feature : {best_feature}")
    print(f"  Weighted F1-score       : {best_score:.4f}\n")

    # Return value useful when called as a library
    best_predictive_feature = {best_feature: best_score}
    return best_predictive_feature


if __name__ == "__main__":
    main()
