from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()

# OpenAI Client

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def generate_ai_insights(df):

    # Dataset Summary

    summary = df.describe(include="all").to_string()

    # Prompt

    prompt = f"""
    Analyze this dataset summary and provide:

    1. Important trends
    2. Business insights
    3. Recommendations
    4. Unusual patterns

    Dataset Summary:

    {summary}
    """

    # OpenAI Response

    response = client.chat.completions.create(
        model="gpt-4.1-mini",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content