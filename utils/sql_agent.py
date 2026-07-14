import duckdb
import pandas as pd


class SQLAgent:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

        duckdb.register(
            "dataset",
            self.df
        )

    # -------------------------

    def execute(self, query: str):

        try:

            result = duckdb.sql(
                query
            ).df()

            return {
                "success": True,
                "data": result,
                "error": None
            }

        except Exception as e:

            return {
                "success": False,
                "data": None,
                "error": str(e)
            }

    # -------------------------

    def sample_queries(self):

        return [

            "SELECT * FROM dataset LIMIT 10",

            "SELECT COUNT(*) AS Total_Rows FROM dataset",

            "SELECT * FROM dataset ORDER BY 1 LIMIT 20"

        ]

    # -------------------------

    def schema(self):

        schema = []

        for col, dtype in zip(
            self.df.columns,
            self.df.dtypes
        ):

            schema.append({

                "Column": col,

                "Datatype": str(dtype)

            })

        return pd.DataFrame(schema)

    # -------------------------

    def describe(self):

        return self.df.describe(
            include="all"
        )

    # -------------------------

    def numeric_columns(self):

        return self.df.select_dtypes(
            include="number"
        ).columns.tolist()

    # -------------------------

    def categorical_columns(self):

        return self.df.select_dtypes(
            include="object"
        ).columns.tolist()