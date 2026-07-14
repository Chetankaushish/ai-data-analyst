import pandas as pd
import plotly.express as px
import streamlit as st


class ChartGenerator:

    def __init__(self, df: pd.DataFrame):

        self.df = df

        self.numeric = df.select_dtypes(
            include="number"
        ).columns.tolist()

        self.text = df.select_dtypes(
            include="object"
        ).columns.tolist()

    # -----------------------------

    def bar_chart(self, x, y):

        fig = px.bar(
            self.df,
            x=x,
            y=y,
            template="plotly_dark",
            text_auto=True
        )

        fig.update_layout(
            height=550
        )

        return fig

    # -----------------------------

    def line_chart(self, x, y):

        fig = px.line(
            self.df,
            x=x,
            y=y,
            template="plotly_dark"
        )

        fig.update_layout(
            height=550
        )

        return fig

    # -----------------------------

    def area_chart(self, x, y):

        fig = px.area(
            self.df,
            x=x,
            y=y,
            template="plotly_dark"
        )

        fig.update_layout(
            height=550
        )

        return fig

    # -----------------------------

    def scatter_chart(self, x, y):

        fig = px.scatter(
            self.df,
            x=x,
            y=y,
            template="plotly_dark"
        )

        fig.update_layout(
            height=550
        )

        return fig

    # -----------------------------

    def histogram(self, column):

        fig = px.histogram(
            self.df,
            x=column,
            template="plotly_dark"
        )

        fig.update_layout(
            height=550
        )

        return fig

    # -----------------------------

    def pie_chart(self, names, values):

        fig = px.pie(
            self.df,
            names=names,
            values=values,
            template="plotly_dark"
        )

        fig.update_layout(
            height=550
        )

        return fig

    # -----------------------------

    def correlation(self):

        if len(self.numeric) < 2:

            return None

        corr = self.df[
            self.numeric
        ].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            template="plotly_dark",
            height=650
        )

        return fig

    # -----------------------------

    def auto_chart(self):

        if len(self.numeric) == 0:

            st.warning(
                "No numeric column found."
            )

            return

        if len(self.text) > 0:

            fig = self.bar_chart(
                self.text[0],
                self.numeric[0]
            )

        else:

            fig = self.histogram(
                self.numeric[0]
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------

    def summary(self):

        return {

            "rows": self.df.shape[0],

            "columns": self.df.shape[1],

            "numeric": len(self.numeric),

            "categorical": len(self.text),

            "missing": int(
                self.df.isnull().sum().sum()
            )

        }