# Importing Dependencies
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import json

st.set_page_config(page_title="Sales Tracker/Predictor", page_icon = "💵")

st.write("This will be an app where I track/predict future sales at AT&T")
def load_transactions(file):
    try:
        df = pd.read_csv(file, header = 4)
        df.columns = [col.strip() for col in df.columns]
        df = df.drop_duplicates()
        df = df.iloc[3:]
        df = df[df['Sales Attain'] != '0.00%']
        df = df.loc[:, ~(df == '-').all()]
        df = df.reset_index(drop=True)
   

        df_compact = df[df['Time Period'] == 'MTD'].copy()

        

        st.write(df_compact)
       
        return df
    except Exception as e:
        st.error(f"Error Processing File: {file}")
        return None

def main():
    uploaded_file = st.file_uploader("Upload your csv", type = ["csv"])

    if uploaded_file is not None:
        df = load_transactions(uploaded_file)
    
main()




