# Advanced Data Analysis App

A powerful data analysis and machine learning web application built with Streamlit. This app allows you to:
- Upload and analyze CSV/Excel files
- Visualize data with various plot types
- Train and evaluate machine learning models
- Analyze feature importance

## Features

- 📊 Data Overview
  - Basic statistics
  - Missing value analysis
  - Data type information
  
- 📈 Data Visualization
  - Correlation matrices
  - Distribution plots
  - Box plots
  
- 🤖 Machine Learning
  - Multiple model options (Linear Regression, Random Forest, XGBoost, CatBoost)
  - Automatic feature preprocessing
  - Model performance metrics
  - Feature importance analysis

## Running Locally

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app_streamlit.py
   ```

## Deploying to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app in Streamlit Cloud:
   - Select your forked repository
   - Set the main file path as `app_streamlit.py`
   - Deploy!

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies 