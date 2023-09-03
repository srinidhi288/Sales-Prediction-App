import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load your dataset (assuming it's already loaded into the 'df' DataFrame)
df = pd.read_csv('Train.csv')

# Define the feature columns and target variable
X = df.drop(columns=["Item_Outlet_Sales"])
y = df["Item_Outlet_Sales"]

# Define categorical and numeric column transformers for preprocessing
categorical_features = ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Location_Type", "Outlet_Type"]
numeric_features = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-hot encoding with handling of unknown categories
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardize numeric variables
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline with preprocessing and linear regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Fit the model
model.fit(X, y)

# Streamlit app
st.title("Linear Regression Sales Prediction App")

st.sidebar.header("Input Features")

# Create input widgets for the user
input_features = {}

for feature in categorical_features + numeric_features:
    if feature in categorical_features:
        # For categorical features, we assume text input
        input_features[feature] = st.sidebar.text_input(f"{feature}:", "Example_Value")
    else:
        # For numeric features, we assume numeric input
        input_features[feature] = st.sidebar.number_input(f"{feature}:", value=0.0, step=0.01)

# When the user clicks the "Predict" button
if st.sidebar.button("Predict"):
    # Create a DataFrame from the user inputs
    user_input = pd.DataFrame([input_features])

    # Use the model to make predictions on the user input
    predicted_sales = model.predict(user_input)

    # Display the prediction to the user
    st.write(f"Predicted Sales: {predicted_sales[0]:.2f}")

# You can add more customization to your Streamlit app as needed

# Optionally, you can add explanations or descriptions to the app using st.write()
st.write("This is a simple Streamlit app for predicting sales using a trained linear regression model.")
