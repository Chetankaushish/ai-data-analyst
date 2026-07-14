import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
from openai import OpenAI
from prophet import Prophet
from sqlalchemy import create_engine
from streamlit_option_menu import option_menu
from components.sidebar import render_sidebar
from components.metrics import render_metrics
from components.dashboard import render_dashboard
from components.chat import render_chat
from utils.sql_agent import SQLAgent
from utils.report import ReportGenerator

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

selected, uploaded_file = render_sidebar()


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

    from components.metrics import render_metrics

    render_metrics(df)

    # Columns

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    text_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    # ---------------- DASHBOARD ----------------

    if selected == "Dashboard":
        
        render_dashboard(df)

    # ---------------- AI CHAT ----------------

    elif selected == "AI Chat":
        render_chat(
            df,
            ask_ai
            )
        
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

    agent = SQLAgent(df)

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
    
    st.markdown("---")
    
    st.subheader("📄 Generate PDF Report")
    
    if st.button("Generate Report"):
        
        with st.spinner("Generating report..."):
            
            try:
                
                insights = ""
                
                report = ReportGenerator(
                    df,
                    insights
                )
                
                
                pdf = report.create_pdf()
                
                st.download_button(
                    
                    "⬇ Download PDF",
                    
                    pdf,
                    
                    "AI_Report.pdf",
                    
                    mime="application/pdf"
                    
                    )
                
                except Exception as e:
                
                st.error(e)

# ---------------- HOME ----------------

else:

    st.info(
        "👈 Upload CSV or Excel file from sidebar"
    )

    st.markdown("""
    ## ✨ By Chetan Sharma
    """)
