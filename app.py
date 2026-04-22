from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_selection import DEFAULT_DATA_PATH, FEATURES, TARGET, evaluate_features, get_best_feature, load_data


@dataclass(frozen=True)
class ModelBundle:
    feature: str
    model: Pipeline
    label_order: list[str]
    crop_feature_means: pd.Series


def _train_single_feature_model(df: pd.DataFrame, feature: str) -> ModelBundle:
    X = df[[feature]]
    y = df[TARGET]

    model: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)

    label_order = list(getattr(model.named_steps["clf"], "classes_", []))
    crop_feature_means = df.groupby(TARGET)[feature].mean().sort_values(ascending=False)

    return ModelBundle(
        feature=feature,
        model=model,
        label_order=label_order,
        crop_feature_means=crop_feature_means,
    )


@st.cache_data(show_spinner=False)
def _load_df(data_path: str) -> pd.DataFrame:
    return load_data(data_path)


@st.cache_data(show_spinner=False)
def _compute_feature_scores(df: pd.DataFrame) -> dict[str, float]:
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return evaluate_features(X_train, X_test, y_train, y_test, FEATURES)


@st.cache_resource(show_spinner=False)
def _get_model_bundle(df: pd.DataFrame, feature: str) -> ModelBundle:
    return _train_single_feature_model(df, feature)


def _percentile(series: pd.Series, value: float) -> float:
    # Simple empirical percentile
    return float((series <= value).mean() * 100.0)


def main() -> None:
    st.set_page_config(page_title="Crop Recommendation (1 soil metric)", page_icon="🌱", layout="centered")

    st.title("🌱 Crop Recommendation — using one soil measurement")
    st.write(
        "This demo shows how well you can predict the best crop when you can only afford to measure **one** soil metric."
    )

    with st.sidebar:
        st.header("Data")
        data_path = st.text_input("CSV path", value=str(DEFAULT_DATA_PATH))
        st.caption("Expected columns: `N`, `P`, `K`, `ph`, `crop`.")

    df = _load_df(data_path)

    st.subheader("1) Pick the single measurement you can afford")
    scores = _compute_feature_scores(df)
    best_feature, best_score = get_best_feature(scores)

    st.write("On this dataset, the best single feature (by weighted F1) is:")
    st.metric(label="Best feature", value=best_feature, delta=f"F1 = {best_score:.4f}")

    feature = st.selectbox(
        "Which soil metric do you have?",
        options=FEATURES,
        index=FEATURES.index(best_feature) if best_feature in FEATURES else 0,
    )

    st.subheader("2) Enter your measurement")
    value = st.number_input(
        f"Your {feature} value",
        value=float(df[feature].median()),
        step=0.1,
        format="%.2f",
    )

    bundle = _get_model_bundle(df, feature)
    proba = bundle.model.predict_proba(pd.DataFrame({feature: [value]}))[0]
    top_idx = np.argsort(proba)[::-1][:3]
    top3 = [(bundle.label_order[i], float(proba[i])) for i in top_idx]

    st.subheader("3) Top crop recommendations")
    for rank, (crop, p) in enumerate(top3, start=1):
        st.write(f"**{rank}. {crop}** — estimated likelihood: **{p:.0%}**")

    st.subheader("Why this makes sense (plain English)")
    pct = _percentile(df[feature], float(value))
    st.write(
        f"Your **{feature} = {value:.2f}** is around the **{pct:.0f}th percentile** compared to the dataset."
    )

    top_crop = top3[0][0]
    crop_mean = float(bundle.crop_feature_means.loc[top_crop])
    overall_mean = float(df[feature].mean())

    direction = "higher" if value >= overall_mean else "lower"
    st.write(
        f"For the top recommendation (**{top_crop}**), the dataset’s average **{feature}** is **{crop_mean:.2f}** "
        f"(overall average is **{overall_mean:.2f}**). Your value is **{direction} than average**, which nudges the model toward those crops."
    )

    with st.expander("Show model score table (single-feature F1)"):
        st.dataframe(
            pd.DataFrame(
                [{"feature": k, "weighted_f1": v} for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Show per-crop average for this feature"):
        st.dataframe(
            bundle.crop_feature_means.rename("mean").reset_index().rename(columns={TARGET: "crop"}),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Note: This is a simple baseline demo (Logistic Regression) intended for explainability, not a production agronomy tool."
    )


if __name__ == "__main__":
    main()

