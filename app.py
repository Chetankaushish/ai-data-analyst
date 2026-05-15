import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
from openai import OpenAI
from prophet import Prophet
from sqlalchemy import create_engine
from streamlit_option_menu import option_menu

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide"
)

# ---------------- OPENROUTER ----------------

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------- DATABASE ----------------

engine = create_engine(
    "sqlite:///data.db"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
    background-color:#0f172a;
    color:white;
}

.main-title{
    font-size:45px;
    font-weight:bold;
    text-align:center;
    color:#38bdf8;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    margin-bottom:30px;
}

div[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #334155;
    padding:15px;
    border-radius:15px;
}

.stButton>button{
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    padding:10px 20px;
    font-weight:bold;
}

section[data-testid="stSidebar"]{
    background:#111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown(
    '<p class="main-title">🚀 AI Data Analyst Pro</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">AI Dashboard using OpenRouter + Streamlit</p>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    selected = option_menu(
        menu_title="Menu",

        options=[
            "Dashboard",
            "AI Chat",
            "Forecasting",
            "SQL Analysis"
        ],

        icons=[
            "bar-chart",
            "robot",
            "graph-up",
            "database"
        ],

        default_index=0
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

# ---------------- FUNCTIONS ----------------

# Clean Data

def clean_data(df):

    df = df.drop_duplicates()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            df[col] = df[col].fillna(
                df[col].mean()
            )

        else:

            df[col] = df[col].astype(str)

            df[col] = df[col].fillna(
                "Unknown"
            )

    return df

# AI Insights

def generate_ai_insights(df):

    summary = df.describe(
        include="all"
    ).to_string()

    prompt = f"""
    Analyze this dataset.

    Give:
    1. Key insights
    2. Trends
    3. Recommendations

    Dataset:
    {summary}
    """

    response = client.chat.completions.create(

        model="deepseek/deepseek-chat-v3-0324",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],

        temperature=0.7
    )

    return response.choices[0].message.content

# AI Chat

def ask_ai(df, question):

    data = df.head(20).to_string()

    prompt = f"""
    Dataset:
    {data}

    Question:
    {question}
    """

    response = client.chat.completions.create(

        model="deepseek/deepseek-chat-v3-0324",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content

# ---------------- MAIN ----------------

if uploaded_file:

    # Read File

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(e)

        st.stop()

    # Clean Data

    df = clean_data(df)

    # Save Database

    df.to_sql(
        "uploaded_data",
        con=engine,
        if_exists="replace",
        index=False
    )

    # Preview

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    # Metrics

    st.subheader("📌 Dashboard Metrics")

    rows = df.shape[0]

    columns = df.shape[1]

    missing = df.isnull().sum().sum()

    numeric_df = df.select_dtypes(
        include="number"
    )

    total = 0

    if not numeric_df.empty:

        total = numeric_df.sum().sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", rows)

    c2.metric("Columns", columns)

    c3.metric("Missing", missing)

    c4.metric(
        "Total",
        f"{total:,.0f}"
    )

    # Columns

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    text_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    # ---------------- DASHBOARD ----------------

    if selected == "Dashboard":

        st.subheader("📊 Charts")

        if numeric_columns and text_columns:

            chart_type = st.selectbox(
                "Chart Type",
                [
                    "Bar Chart",
                    "Line Chart",
                    "Pie Chart",
                    "Scatter Plot"
                ]
            )

            x_col = st.selectbox(
                "Select X Axis",
                text_columns
            )

            y_col = st.selectbox(
                "Select Y Axis",
                numeric_columns
            )

            # BAR

            if chart_type == "Bar Chart":

                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    template="plotly_dark"
                )

            # LINE

            elif chart_type == "Line Chart":

                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    template="plotly_dark"
                )

            # PIE

            elif chart_type == "Pie Chart":

                fig = px.pie(
                    df,
                    names=x_col,
                    values=y_col,
                    template="plotly_dark"
                )

            # SCATTER

            else:

                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    template="plotly_dark"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Heatmap

        if len(numeric_columns) >= 2:

            st.subheader("🔥 Correlation Heatmap")

            corr = df[
                numeric_columns
            ].corr()

            heatmap = px.imshow(
                corr,
                text_auto=True,
                template="plotly_dark"
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True
            )

        # AI Insights

        st.subheader("🤖 AI Insights")

        if st.button("Generate Insights"):

            with st.spinner("Analyzing..."):

                insights = generate_ai_insights(df)

                st.success("Done")

                st.write(insights)

    # ---------------- AI CHAT ----------------

    elif selected == "AI Chat":

        st.subheader("💬 Chat with Data")

        question = st.text_input(
            "Ask Question"
        )

        if st.button("Ask AI"):

            if question:

                with st.spinner("Thinking..."):

                    answer = ask_ai(
                        df,
                        question
                    )

                    st.write(answer)

    # ---------------- FORECASTING ----------------

    elif selected == "Forecasting":

        st.subheader("📈 Forecasting")

        st.info(
            "Dataset must contain date column."
        )

        try:

            date_columns = df.select_dtypes(
                include=["datetime64"]
            ).columns.tolist()

            if date_columns and numeric_columns:

                date_col = st.selectbox(
                    "Date Column",
                    date_columns
                )

                value_col = st.selectbox(
                    "Value Column",
                    numeric_columns
                )

                forecast_days = st.slider(
                    "Forecast Days",
                    7,
                    365,
                    30
                )

                forecast_df = df[
                    [date_col, value_col]
                ]

                forecast_df.columns = [
                    "ds",
                    "y"
                ]

                model = Prophet()

                model.fit(forecast_df)

                future = model.make_future_dataframe(
                    periods=forecast_days
                )

                forecast = model.predict(
                    future
                )

                fig = px.line(
                    forecast,
                    x="ds",
                    y="yhat",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    "No date column found"
                )

        except Exception as e:

            st.error(e)

    # ---------------- SQL ANALYSIS ----------------

    elif selected == "SQL Analysis":

        st.subheader("🗄 SQL Query")

        sql_query = st.text_area(
            "Write SQL Query",
            "SELECT * FROM df LIMIT 5"
        )

        if st.button("Run SQL"):

            try:

                result = duckdb.query(
                    sql_query
                ).to_df()

                st.dataframe(result)

            except Exception as e:

                st.error(e)

    # ---------------- DOWNLOAD ----------------

    st.subheader("⬇ Download Data")

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )

# ---------------- HOME ----------------

else:

    st.info(
        "👈 Upload CSV or Excel file from sidebar"
    )

    st.markdown("""
    ## ✨ By Chetan Sharma
    """)
