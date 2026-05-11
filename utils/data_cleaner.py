import pandas as pd

def clean_data(df):

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing values

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = df[column].fillna("Unknown")

        else:

            df[column] = df[column].fillna(df[column].mean())

    return df