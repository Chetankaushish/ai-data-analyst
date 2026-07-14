import streamlit as st
import plotly.express as px
import pandas as pd


def render_dashboard(df: pd.DataFrame):

    st.markdown("## 📊 Smart Dashboard")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    text_columns = df.select_dtypes(include="object").columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns found.")
        return

    col1, col2 = st.columns(2)

    with col1:

        chart_type = st.selectbox(
            "📈 Chart Type",
            [
                "Bar",
                "Line",
                "Area",
                "Scatter",
                "Histogram",
                "Box",
                "Violin",
                "Pie"
            ]
        )

    with col2:

        y_col = st.selectbox(
            "📊 Numeric Column",
            numeric_columns
        )

    x_col = None

    if text_columns:
        x_col = st.selectbox(
            "📂 Category",
            text_columns
        )

    fig = None

    if chart_type == "Bar" and x_col:
        fig = px.bar(df, x=x_col, y=y_col)

    elif chart_type == "Line" and x_col:
        fig = px.line(df, x=x_col, y=y_col)

    elif chart_type == "Area" and x_col:
        fig = px.area(df, x=x_col, y=y_col)

    elif chart_type == "Scatter" and x_col:
        fig = px.scatter(df, x=x_col, y=y_col)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=y_col)

    elif chart_type == "Box":
        fig = px.box(df, y=y_col)

    elif chart_type == "Violin":
        fig = px.violin(df, y=y_col)

    elif chart_type == "Pie" and x_col:
        fig = px.pie(df, names=x_col, values=y_col)

    if fig:

        fig.update_layout(
            template="plotly_dark",
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    if len(numeric_columns) >= 2:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_columns].corr()

        heatmap = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues"
        )

        heatmap.update_layout(
            template="plotly_dark",
            height=650
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )