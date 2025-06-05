import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utils import load_transactions


def clean_data(df):
    df = df.copy()

    # Log column names for debugging
    if df.empty or df.columns.size == 0:
        raise ValueError("DataFrame has no columns.")

    first_col = df.columns[0]
    if not df[first_col].dtype == object:
        raise ValueError("First column is not string-type for STORE filtering.")

    df = df[df[first_col].str.contains("STORE", na=False)]
    df.reset_index(drop=True, inplace=True)

    # Continue with numeric cleaning as before
    to_clean = [
        'Sales Attain',
        'Trade-In Opp Att Rt',
        'HTP Tot Feat Att Rt',
        'NextUp AT Installment Plan Mix',
        'ProtAdv + HTP Tot Att Rt',
        'Overall CSAT'
    ]

    for col in to_clean:
        if col in df.columns:
            df[col] = df[col].replace('-', np.nan).str.rstrip('%').astype(float)
        else:
            raise ValueError(f"Missing expected column: {col}")

    return df

def preprocess_data(df):


    selected_features = [
        'Trade-In Opp Att Rt',
        'HTP Tot Feat Att Rt',
        'NextUp AT Installment Plan Mix',
        'ProtAdv + HTP Tot Att Rt',
        'Overall CSAT',
        'Sales Attain'
    ]

    df = df[selected_features]
    df = df.dropna(axis=1, how='all')

    # Impute missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Check if we still have data
    if df.empty:
        raise ValueError("No rows left after imputation.")

    X = df.drop(columns=['Sales Attain'])
    y = df['Sales Attain']

    if X.shape[0] == 0:
        raise ValueError("No valid samples found for scaling.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['Sales Attain'] = y.values

    return df_scaled


def create_lag_features(df, group_by_col=None):
    df = df.copy()

    # Optionally group by store/rep if your data spans multiple
    if group_by_col:
        df = df.sort_values(by=[group_by_col, 'Date'])  # Ensure sorted
    else:
        df = df.sort_values(by='Date')

    # Create lag features for past Sales Attain and other predictors
    lag_features = ['Sales Attain', 'Trade-In Opp Att Rt', 'HTP Tot Feat Att Rt']
    
    for feature in lag_features:
        df[f'{feature} Lag1'] = df.groupby(group_by_col)[feature].shift(1) if group_by_col else df[feature].shift(1)

    # Drop rows with NaNs introduced by lagging
    df = df.dropna()

    return df


def train_model(df):
    # Separate features and target
    X = df.drop(columns=['Sales Attain'])
    y = df['Sales Attain']

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model, X_test, y_test, y_pred


