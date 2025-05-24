# Importing Dependencies
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import json
from utils import load_transactions 

st.set_page_config(page_title="Sales Tracker/Predictor", page_icon = "📊")

def main():
    uploaded_file = st.file_uploader("Upload your csv", type=["csv"])
    try:
        if uploaded_file is not None:
            df = load_transactions(uploaded_file)
            st.write(df)  # View the cleaned df
            return df
    except Exception as e:
        st.write(f"Error loading in file: {uploaded_file}")
        return None
    
# Only run the main function if this file is being run directly
if __name__ == "__main__":
    main()
