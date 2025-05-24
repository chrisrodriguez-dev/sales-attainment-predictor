import pandas as pd

def load_transactions(file):
    df = pd.read_csv(file, header=4)
    df.columns = [col.strip() for col in df.columns]
    df = df.drop_duplicates()
    df = df.iloc[3:]
    df = df[df['Sales Attain'] != '0.00%']
    df = df.loc[:, ~(df == '-').all()]
    df = df.reset_index(drop=True)
    df_compact = df[df['Time Period'] == 'MTD'].copy()
    return df_compact