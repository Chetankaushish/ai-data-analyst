from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()

# OpenRouter Client

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def generate_ai_insights(df):

    # Dataset Summary

    summary = df.describe(include="all").to_string()

    # Prompt

    prompt = f"""
    Analyze this dataset and provide:

    1. Important business insights
    2. Key trends
    3. Recommendations
    4. Unusual patterns
    5. Data quality observations

    Dataset Summary:

    {summary}
    """

    # AI Response

    response = client.chat.completions.create(

        model="deepseek/deepseek-chat-v3-0324",

        messages=[
            {
                "role": "system",
                "content": "You are a professional AI Data Analyst."
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
