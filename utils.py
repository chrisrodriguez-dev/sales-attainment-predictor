import pandas as pd
import streamlit as st
def load_transactions(file):
    try:
        df = pd.read_csv(file, header=4)
        df.columns = [col.strip() for col in df.columns]
        df = df.drop_duplicates()
        df = df.iloc[3:]

        if 'Sales Attain' not in df.columns:
            st.warning("⚠️ 'Sales Attain' column not found.")
            return None

        df = df[df['Sales Attain'] != '0.00%']
        df = df.loc[:, ~(df == '-').all()]
        df = df.reset_index(drop=True)

        if 'Time Period' in df.columns:
            df = df[df['Time Period'] == 'MTD'].copy()

        if df.empty:
            st.warning("⚠️ DataFrame is empty after preprocessing.")
            return None

        return df

    except Exception as e:
        st.error(f"❌ Failed to load transactions: {e}")
        return None
