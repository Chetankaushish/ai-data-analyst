import streamlit as st
from utils.report import ReportGenerator


def render_report_page(df, insights=""):

    st.header("📄 AI Report Generator")

    st.info(
        "Generate a professional PDF report of your dataset."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing", int(df.isnull().sum().sum()))

    st.divider()

    if st.button(
        "📄 Generate Professional Report",
        use_container_width=True
    ):

        with st.spinner("Generating Report..."):

            try:

                report = ReportGenerator(
                    df=df,
                    insights=insights
                )

                pdf = report.create_pdf()

                st.success(
                    "Report Generated Successfully"
                )

                st.download_button(

                    "⬇ Download PDF",

                    pdf,

                    file_name="AI_Data_Analyst_Report.pdf",

                    mime="application/pdf",

                    use_container_width=True

                )

            except Exception as e:

                st.error(e)

    st.divider()

    st.subheader("Preview")

    st.dataframe(
        df.head(15),
        use_container_width=True
    )