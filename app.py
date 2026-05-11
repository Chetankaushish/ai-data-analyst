import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_cleaner import clean_data
from utils.ai_insights import generate_ai_insights

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    font-size: 40px;
    font-weight: bold;
    color: #00E5FF;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown('<p class="title">🚀 AI Data Analyst Pro</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Power BI Style Dashboard with AI Insights</p>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# ---------------- MAIN APP ----------------

if uploaded_file:

    # Read file

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    # Clean data

    df = clean_data(df)

    # ---------------- DATA PREVIEW ----------------

    st.subheader("📄 Dataset Preview")

    st.dataframe(df, use_container_width=True)

    # ---------------- KPI SECTION ----------------

    st.subheader("📌 Dashboard Metrics")

    rows = df.shape[0]
    columns = df.shape[1]
    missing = df.isnull().sum().sum()

    numeric_df = df.select_dtypes(include="number")

    total_value = 0

    if not numeric_df.empty:
        total_value = numeric_df.sum().sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", rows)
    col2.metric("Columns", columns)
    col3.metric("Missing Values", missing)
    col4.metric("Numeric Total", f"{total_value:,.0f}")

    # ---------------- CHART SECTION ----------------

    st.subheader("📈 Interactive Visualization")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    if numeric_columns and categorical_columns:

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Bar Chart",
                "Line Chart",
                "Pie Chart",
                "Scatter Plot"
            ]
        )

        x_axis = st.selectbox("Select X-Axis", categorical_columns)

        y_axis = st.selectbox("Select Y-Axis", numeric_columns)

        # ---------------- CHARTS ----------------

        if chart_type == "Bar Chart":

            fig = px.bar(
                df,
                x=x_axis,
                y=y_axis,
                template="plotly_dark",
                text_auto=True
            )

        elif chart_type == "Line Chart":

            fig = px.line(
                df,
                x=x_axis,
                y=y_axis,
                template="plotly_dark",
                markers=True
            )

        elif chart_type == "Pie Chart":

            fig = px.pie(
                df,
                names=x_axis,
                values=y_axis,
                hole=0.5,
                template="plotly_dark"
            )

        else:

            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                size=y_axis,
                template="plotly_dark"
            )

        fig.update_layout(
            height=550,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- CORRELATION HEATMAP ----------------

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
            height=600
        )

        st.plotly_chart(heatmap, use_container_width=True)

    # ---------------- AI INSIGHTS ----------------

    st.subheader("🤖 AI Insights")

    if st.button("Generate AI Insights"):

        with st.spinner("Analyzing Data with AI..."):

            insights = generate_ai_insights(df)

            st.success("Analysis Complete")

            st.write(insights)

    # ---------------- DOWNLOAD CSV ----------------

    st.subheader("⬇ Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# ---------------- HOME SCREEN ----------------

else:

    st.info("👈 Upload a CSV or Excel file from the sidebar to start.")

    st.markdown("""
    ### ✨ Features

    - AI Generated Insights
    - Power BI Style Dashboard
    - Interactive Charts
    - KPI Metrics
    - Correlation Heatmaps
    - Download Cleaned Dataset
    """)