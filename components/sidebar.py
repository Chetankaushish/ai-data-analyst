import streamlit as st
from streamlit_option_menu import option_menu


def render_sidebar():
    """
    Render Premium Sidebar
    Returns:
        selected_menu, uploaded_file
    """

    with st.sidebar:

        st.markdown(
            """
            <div style="text-align:center;padding:10px;">
                <h2 style="color:#60A5FA;">🤖 AI Data Analyst</h2>
                <p style="color:#94A3B8;font-size:14px;">
                    AI Powered Business Intelligence
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected = option_menu(
            menu_title=None,
            options=[
                "Dashboard",
                "AI Chat",
                "Forecasting",
                "SQL Analysis",
                "Data Cleaning",
                "Reports",
                "Settings"
            ],
            icons=[
                "speedometer2",
                "robot",
                "graph-up-arrow",
                "database-fill",
                "magic",
                "file-earmark-text",
                "gear-fill"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "8px",
                    "background-color": "#0B1120"
                },
                "icon": {
                    "color": "#60A5FA",
                    "font-size": "18px"
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "4px",
                    "border-radius": "12px",
                    "--hover-color": "#1E293B",
                },
                "nav-link-selected": {
                    "background-color": "#2563EB",
                    "color": "white",
                },
            },
        )

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "📂 Upload Dataset",
            type=["csv", "xlsx"],
            help="Supported formats: CSV & Excel"
        )

        st.markdown("---")

        st.markdown(
            """
            ### 📌 Quick Tips

            ✅ Upload CSV/Excel

            🤖 Ask AI Questions

            📊 Generate Dashboard

            📈 Forecast Trends

            📄 Export Reports
            """
        )

        st.markdown("---")

        st.caption("🚀 AI Data Analyst Pro v2.0")

    return selected, uploaded_file