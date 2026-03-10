import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values="?")
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        )
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().str.lower()
    return df