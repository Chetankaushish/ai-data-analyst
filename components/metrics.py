import streamlit as st
import pandas as pd


def calculate_quality_score(df: pd.DataFrame) -> int:
    """
    Calculate a simple data quality score (0-100)
    """

    score = 100

    missing_pct = (
        df.isnull().sum().sum()
        / (df.shape[0] * df.shape[1])
    ) * 100

    duplicate_pct = (
        df.duplicated().sum()
        / max(len(df), 1)
    ) * 100

    score -= missing_pct * 0.8
    score -= duplicate_pct * 0.5

    score = max(0, min(100, round(score)))

    return score


def render_metrics(df: pd.DataFrame):

    rows = len(df)
    cols = len(df.columns)

    missing = int(df.isnull().sum().sum())

    duplicates = int(df.duplicated().sum())

    numeric = len(
        df.select_dtypes(include="number").columns
    )

    categorical = len(
        df.select_dtypes(include="object").columns
    )

    memory = (
        df.memory_usage(deep=True).sum()
        / 1024
        / 1024
    )

    quality = calculate_quality_score(df)

    st.markdown("## 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📄 Rows",
        f"{rows:,}"
    )

    c2.metric(
        "📑 Columns",
        cols
    )

    c3.metric(
        "❌ Missing",
        missing
    )

    c4.metric(
        "♻ Duplicates",
        duplicates
    )

    st.markdown("")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "🔢 Numeric",
        numeric
    )

    c6.metric(
        "🔤 Categorical",
        categorical
    )

    c7.metric(
        "💾 Memory",
        f"{memory:.2f} MB"
    )

    c8.metric(
        "⭐ Quality",
        f"{quality}%"
    )

    st.progress(quality / 100)

    if quality >= 90:

        st.success("Excellent dataset quality.")

    elif quality >= 70:

        st.info("Good dataset quality.")

    elif quality >= 50:

        st.warning("Dataset needs cleaning.")

    else:

        st.error("Poor data quality detected.")

    return {
        "rows": rows,
        "columns": cols,
        "missing": missing,
        "duplicates": duplicates,
        "numeric": numeric,
        "categorical": categorical,
        "memory": memory,
        "quality": quality,
    }