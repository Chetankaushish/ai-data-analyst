import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
from prophet import Prophet
from openai import OpenAI
from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh
from sqlalchemy import create_engine
import streamlit_authenticator as stauth
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
import hashlib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide"
)

# ---------------- REAL TIME REFRESH ----------------

st_autorefresh(
    interval=5000,
    key="dashboard_refresh"
)

# ---------------- DATABASE ----------------

engine = create_engine(
    "sqlite:///analytics.db"
)

# ---------------- OPENROUTER CLIENT ----------------

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------- LOGIN SYSTEM ----------------

users = {

    "admin": "12345",

    "chetan": "password",

    "guest": "guest123"
}

names = list(users.keys())

usernames = list(users.keys())

passwords = list(users.values())

hashed_passwords = stauth.Hasher(
    passwords
).generate()

authenticator = stauth.Authenticate(
    names,
    usernames,
    hashed_passwords,
    "ai_dashboard",
    "abcdef",
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login(
    "Login",
    "main"
)

if authentication_status == False:

    st.error("Incorrect Username or Password")

if authentication_status == None:

    st.warning("Please Enter Username and Password")

# ---------------- MAIN APP ----------------

if authentication_status:

    authenticator.logout("Logout", "sidebar")

    st.sidebar.success(f"Welcome {name}")

    # ---------------- DARK MODE ----------------

    theme = st.sidebar.toggle(
        "Dark Mode",
        value=True
    )

    if theme:

        template_style = "plotly_dark"

        background = "#0E1117"

    else:

        template_style = "plotly"

        background = "#FFFFFF"

    # ---------------- CUSTOM CSS ----------------

    st.markdown(f"""
    <style>

    .main {{
        background-color: {background};
    }}

    .title {{
        font-size: 42px;
        font-weight: bold;
        color: #00E5FF;
        text-align: center;
    }}

    .subtitle {{
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }}

    div[data-testid="metric-container"] {{
        border: 1px solid #333;
        padding: 15px;
        border-radius: 15px;
    }}

    </style>
    """, unsafe_allow_html=True)

    # ---------------- HEADER ----------------

    st.markdown(
        '<p class="title">🚀 AI Data Analyst Pro</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Enterprise AI Analytics Platform</p>',
        unsafe_allow_html=True
    )

    # ---------------- SIDEBAR ----------------

    with st.sidebar:

        selected = option_menu(

            menu_title="AI Analytics",

            options=[
                "Dashboard",
                "AI Chat",
                "Forecasting",
                "SQL Analysis",
                "SQL Chatbot"
            ],

            icons=[
                "bar-chart",
                "robot",
                "graph-up",
                "database",
                "chat-dots"
            ],

            default_index=0
        )

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel File",
            type=["csv", "xlsx"]
        )

    # ---------------- FUNCTIONS ----------------

    def clean_data(df):

        df = df.drop_duplicates()

        for column in df.columns:

            if df[column].dtype == "object":

                df[column] = df[column].fillna(
                    "Unknown"
                )

            else:

                df[column] = df[column].fillna(
                    df[column].mean()
                )

        return df

    def generate_ai_insights(df):

        summary = df.describe(
            include="all"
        ).to_string()

        prompt = f"""
        Analyze this dataset and provide:

        1. Business insights
        2. Key trends
        3. Recommendations
        4. Unusual patterns

        Dataset Summary:

        {summary}
        """

        response = client.chat.completions.create(

            model="deepseek/deepseek-chat-v3-0324",

            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI Data Analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def ask_ai(df, question):

        preview = df.head(20).to_string()

        prompt = f"""
        Dataset Preview:

        {preview}

        User Question:

        {question}

        Give a professional business answer.
        """

        response = client.chat.completions.create(

            model="deepseek/deepseek-chat-v3-0324",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def create_pdf_report(text):

        doc = SimpleDocTemplate(
            "AI_Report.pdf"
        )

        styles = getSampleStyleSheet()

        story = []

        title = Paragraph(
            "AI Analytics Report",
            styles['Title']
        )

        body = Paragraph(
            text,
            styles['BodyText']
        )

        story.append(title)

        story.append(Spacer(1, 20))

        story.append(body)

        doc.build(story)

    # ---------------- FILE UPLOAD ----------------

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        # Clean Data

        df = clean_data(df)

        # Save To Database

        df.to_sql(
            "uploaded_data",
            con=engine,
            if_exists="replace",
            index=False
        )

        # Dataset Preview

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        # KPI Metrics

        st.subheader("📌 Dashboard Metrics")

        rows = df.shape[0]

        columns = df.shape[1]

        missing = df.isnull().sum().sum()

        numeric_df = df.select_dtypes(
            include="number"
        )

        total_value = 0

        if not numeric_df.empty:

            total_value = numeric_df.sum().sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", rows)

        col2.metric("Columns", columns)

        col3.metric("Missing", missing)

        col4.metric(
            "Numeric Total",
            f"{total_value:,.0f}"
        )

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include="object"
        ).columns.tolist()

        # ---------------- DASHBOARD ----------------

        if selected == "Dashboard":

            st.subheader(
                "📈 Interactive Visualization"
            )

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

                x_axis = st.selectbox(
                    "Select X-Axis",
                    categorical_columns
                )

                y_axis = st.selectbox(
                    "Select Y-Axis",
                    numeric_columns
                )

                if chart_type == "Bar Chart":

                    fig = px.bar(
                        df,
                        x=x_axis,
                        y=y_axis,
                        template=template_style,
                        text_auto=True
                    )

                elif chart_type == "Line Chart":

                    fig = px.line(
                        df,
                        x=x_axis,
                        y=y_axis,
                        template=template_style,
                        markers=True
                    )

                elif chart_type == "Pie Chart":

                    fig = px.pie(
                        df,
                        names=x_axis,
                        values=y_axis,
                        hole=0.5,
                        template=template_style
                    )

                else:

                    fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        size=y_axis,
                        template=template_style
                    )

                fig.update_layout(height=550)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # Auto Dashboard

            st.subheader("⚡ Auto Dashboard")

            if numeric_columns and categorical_columns:

                auto_x = categorical_columns[0]

                auto_y = numeric_columns[0]

                c1, c2 = st.columns(2)

                with c1:

                    auto_bar = px.bar(
                        df,
                        x=auto_x,
                        y=auto_y,
                        template=template_style
                    )

                    st.plotly_chart(
                        auto_bar,
                        use_container_width=True
                    )

                with c2:

                    auto_line = px.line(
                        df,
                        x=auto_x,
                        y=auto_y,
                        template=template_style
                    )

                    st.plotly_chart(
                        auto_line,
                        use_container_width=True
                    )

            # Heatmap

            if len(numeric_columns) >= 2:

                st.subheader(
                    "🔥 Correlation Heatmap"
                )

                corr = df[
                    numeric_columns
                ].corr()

                heatmap = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="Blues"
                )

                heatmap.update_layout(
                    template=template_style
                )

                st.plotly_chart(
                    heatmap,
                    use_container_width=True
                )

            # AI Insights

            st.subheader("🤖 AI Insights")

            if st.button(
                "Generate AI Insights"
            ):

                with st.spinner(
                    "Analyzing Dataset..."
                ):

                    insights = generate_ai_insights(
                        df
                    )

                    st.success(
                        "Analysis Complete"
                    )

                    st.write(insights)

                    create_pdf_report(
                        insights
                    )

                    with open(
                        "AI_Report.pdf",
                        "rb"
                    ) as file:

                        st.download_button(
                            label="Download PDF Report",
                            data=file,
                            file_name="AI_Report.pdf",
                            mime="application/pdf"
                        )

        # ---------------- AI CHAT ----------------

        elif selected == "AI Chat":

            st.subheader(
                "💬 Chat With Your Data"
            )

            question = st.text_input(
                "Ask anything about your dataset"
            )

            if st.button("Ask AI"):

                if question:

                    with st.spinner(
                        "AI Thinking..."
                    ):

                        answer = ask_ai(
                            df,
                            question
                        )

                        st.success(
                            "Answer Generated"
                        )

                        st.write(answer)

        # ---------------- FORECASTING ----------------

        elif selected == "Forecasting":

            st.subheader(
                "📈 AI Forecasting"
            )

            date_columns = df.select_dtypes(
                include=["datetime64"]
            ).columns.tolist()

            if (
                date_columns
                and numeric_columns
            ):

                selected_date = st.selectbox(
                    "Select Date Column",
                    date_columns
                )

                selected_target = st.selectbox(
                    "Select Forecast Column",
                    numeric_columns
                )

                forecast_days = st.slider(
                    "Forecast Days",
                    7,
                    365,
                    30
                )

                forecast_df = df[
                    [
                        selected_date,
                        selected_target
                    ]
                ]

                forecast_df.columns = [
                    "ds",
                    "y"
                ]

                model = Prophet()

                model.fit(
                    forecast_df
                )

                future = model.make_future_dataframe(
                    periods=forecast_days
                )

                forecast = model.predict(
                    future
                )

                forecast_chart = px.line(
                    forecast,
                    x="ds",
                    y="yhat",
                    template=template_style
                )

                st.plotly_chart(
                    forecast_chart,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Dataset must contain date column."
                )

        # ---------------- SQL ANALYSIS ----------------

        elif selected == "SQL Analysis":

            st.subheader(
                "🗄 SQL Query System"
            )

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

                    st.error(str(e))

        # ---------------- SQL CHATBOT ----------------

        elif selected == "SQL Chatbot":

            st.subheader(
                "🤖 SQL Chatbot"
            )

            sql_question = st.text_input(
                "Ask SQL Question"
            )

            if st.button(
                "Generate SQL"
            ):

                sql_prompt = f"""
                Convert this question into SQL.

                Question:
                {sql_question}

                Table name is df.
                """

                response = client.chat.completions.create(

                    model="deepseek/deepseek-chat-v3-0324",

                    messages=[
                        {
                            "role": "user",
                            "content": sql_prompt
                        }
                    ]
                )

                generated_sql = response.choices[
                    0
                ].message.content

                st.code(generated_sql)

                try:

                    result = duckdb.query(
                        generated_sql
                    ).to_df()

                    st.dataframe(result)

                except Exception as e:

                    st.error(str(e))

        # Download CSV

        st.subheader(
            "⬇ Download Cleaned Dataset"
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "👈 Upload a CSV or Excel file from the sidebar."
        )

        st.markdown("""
        ## ✨By Chetan Sharma
        """)
