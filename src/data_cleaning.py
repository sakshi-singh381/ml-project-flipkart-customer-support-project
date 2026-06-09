import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    df = df.drop_duplicates()
    return df