import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import shap

# Set page config
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stButton>button {
        background-color: #7E57C2;
        color: white;
    }
    .stButton>button:hover {
        background-color: #9575CD;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title
st.title("📊 Advanced Data Analysis Tool")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Keep original data
        original_df = df.copy()
        
        # Sidebar for analysis options
        st.sidebar.title("Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Data Overview", "Visualization", "Machine Learning"]
        )
        
        if analysis_type == "Data Overview":
            st.header("📋 Data Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", df.shape)
                st.write("Number of Missing Values:", df.isnull().sum().sum())
            
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
        elif analysis_type == "Visualization":
            st.header("📈 Data Visualization")
            
            # Select columns for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            viz_type = st.selectbox(
                "Choose Visualization Type",
                ["Correlation Matrix", "Distribution Plots", "Box Plots"]
            )
            
            if viz_type == "Correlation Matrix":
                plt.figure(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
                st.pyplot(plt)
                
            elif viz_type == "Distribution Plots":
                selected_col = st.selectbox("Select Column", numeric_cols)
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=selected_col, kde=True)
                st.pyplot(plt)
                
            elif viz_type == "Box Plots":
                selected_col = st.selectbox("Select Column", numeric_cols)
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, y=selected_col)
                st.pyplot(plt)
                
        elif analysis_type == "Machine Learning":
            st.header("🤖 Machine Learning")
            
            # Select target variable
            target_col = st.selectbox("Select Target Variable", df.columns)
            
            # Select features
            feature_cols = st.multiselect("Select Features", 
                                        [col for col in df.columns if col != target_col],
                                        default=[col for col in df.columns if col != target_col])
            
            if len(feature_cols) > 0:
                # Prepare data
                X = df[feature_cols]
                y = df[target_col]
                
                # Handle categorical variables
                X = pd.get_dummies(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Select model
                model_type = st.selectbox(
                    "Select Model Type",
                    ["Linear Regression", "Random Forest", "XGBoost", "CatBoost"]
                )
                
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        if model_type == "Linear Regression":
                            model = LinearRegression()
                        elif model_type == "Random Forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == "XGBoost":
                            model = XGBRegressor(random_state=42)
                        else:
                            model = CatBoostRegressor(random_state=42, verbose=False)
                        
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Display results
                        st.subheader("Model Performance")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                        with col2:
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        # Feature importance
                        if model_type in ["Random Forest", "XGBoost", "CatBoost"]:
                            st.subheader("Feature Importance")
                            importances = pd.DataFrame({
                                'feature': X.columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            plt.figure(figsize=(10, 6))
                            sns.barplot(data=importances.head(10), x='importance', y='feature')
                            plt.title("Top 10 Most Important Features")
                            st.pyplot(plt)
                            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a data file to begin analysis.") 