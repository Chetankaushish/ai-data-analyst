import streamlit as st


def render_settings():

    st.title("⚙️ Settings")

    st.markdown("Manage AI Data Analyst settings")

    st.divider()

    # ---------------- AI ---------------- #

    st.subheader("🤖 AI Configuration")

    model = st.selectbox(
        "AI Model",
        [
            "DeepSeek Chat v3",
            "GPT-4.1",
            "Claude Sonnet",
            "Gemini 2.5 Flash"
        ]
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.3,
        0.1
    )

    max_tokens = st.slider(
        "Max Tokens",
        256,
        4096,
        1024
    )

    st.divider()

    # ---------------- Dashboard ---------------- #

    st.subheader("📊 Dashboard")

    default_chart = st.selectbox(
        "Default Chart",
        [
            "Bar Chart",
            "Line Chart",
            "Area Chart",
            "Scatter Plot"
        ]
    )

    show_kpi = st.toggle(
        "Show KPI Cards",
        value=True
    )

    auto_insights = st.toggle(
        "Generate AI Insights Automatically",
        value=True
    )

    st.divider()

    # ---------------- Theme ---------------- #

    st.subheader("🎨 Appearance")

    theme = st.radio(
        "Theme",
        [
            "Dark",
            "Light"
        ],
        horizontal=True
    )

    compact = st.toggle(
        "Compact Mode",
        value=False
    )

    st.divider()

    # ---------------- API ---------------- #

    st.subheader("🔑 API Status")

    if "OPENROUTER_API_KEY" in st.secrets:

        st.success("OpenRouter API Connected ✅")

    else:

        st.error("API Key Missing ❌")

    st.divider()

    # ---------------- About ---------------- #

    st.subheader("ℹ About")

    st.info(
        """
AI Data Analyst Pro

Version : 2.0

Developer : Chetan Sharma

Built with Streamlit + OpenRouter + Plotly
"""
    )

    st.divider()

    if st.button(
        "💾 Save Settings",
        use_container_width=True
    ):

        st.success(
            "Settings saved successfully."
        )

    return {

        "model": model,

        "temperature": temperature,

        "max_tokens": max_tokens,

        "default_chart": default_chart,

        "show_kpi": show_kpi,

        "auto_insights": auto_insights,

        "theme": theme,

        "compact": compact

    }