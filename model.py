from main import load_transactions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_transactions


def clean_data(df):
    df['Sales Attain'] = df['Sales Attain'].str.rstrip('%').astype(float)
    df['Trade-In Opp Att Rt'] = df['Trade-In Opp Att Rt'].str.rstrip('%').astype(float)
    df['Overall CSAT'] = df['Overall CSAT'].astype(float)

#imputing/eliminating and encoding features
def preprocess_data(df):
    pass

def create_lag_features(df):
    pass

def train_model(df):
    pass



