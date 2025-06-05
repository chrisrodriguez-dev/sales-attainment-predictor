# Importing Dependencies
import streamlit as st
import plotly.express as px
from model import clean_data, preprocess_data, create_lag_features, train_model
from utils import load_transactions 

st.set_page_config(page_title="Sales Tracker/Predictor", page_icon = "📊")

def main():
    st.title("📊 Sales Attain Forecasting")

    uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            raw_df = load_transactions(uploaded_file)
            if raw_df is not None:
                df_clean = clean_data(raw_df)
            else:
                st.stop()

            st.subheader("📋 Raw Uploaded Data")
            st.write(raw_df)

            # Run pipeline
            df_clean = clean_data(raw_df)
            df_preprocessed = preprocess_data(df_clean)
            df_lagged = create_lag_features(df_preprocessed)
            model, X_test, y_test, y_pred = train_model(df_lagged)

            # Show sample of final training data
            st.subheader("🧼 Cleaned + Feature-Engineered Data")
            st.dataframe(df_lagged)

            # Show metrics
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("📈 Model Evaluation Metrics")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("R²", f"{r2:.2f}")

            # Plot Actual vs Predicted
            st.subheader("🔍 Actual vs Predicted")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Sales Attain')
            ax.set_ylabel('Predicted Sales Attain')
            ax.set_title('Actual vs Predicted Sales Attain')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

# Only run the main function if this file is being run directly
if __name__ == "__main__":
    main()
