from openai import OpenAI
import pandas as pd


class AIEngine:

    def __init__(self, api_key):

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.model = "deepseek/deepseek-chat-v3-0324"

    # -----------------------------------------

    def dataset_summary(self, df: pd.DataFrame):

        return f"""
Rows : {len(df)}

Columns : {len(df.columns)}

Column Names :

{', '.join(df.columns)}

Data Types :

{df.dtypes.to_string()}

Missing Values :

{df.isnull().sum().to_string()}

Statistics :

{df.describe(include='all').to_string()}
"""

    # -----------------------------------------

    def ask(self, df, question):

        prompt = f"""
You are a Senior Data Analyst.

Analyze the following dataset.

{self.dataset_summary(df)}

User Question:

{question}

Rules:

1. Answer professionally.

2. Give business insights.

3. Keep answer concise.

4. Suggest recommendations whenever possible.

5. If suitable, recommend visualization.
"""

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4

        )

        return response.choices[0].message.content

    # -----------------------------------------

    def executive_summary(self, df):

        prompt = f"""
Create an executive summary.

Dataset

{self.dataset_summary(df)}

Return:

Executive Summary

Important Trends

Risk

Recommendations

Opportunities
"""

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return response.choices[0].message.content

    # -----------------------------------------

    def business_insights(self, df):

        prompt = f"""
Analyze this business dataset.

{self.dataset_summary(df)}

Generate:

Top Insights

Hidden Trends

Anomalies

Future Suggestions
"""

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return response.choices[0].message.content