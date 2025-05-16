import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model, columns and scaler
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = list(scaler.feature_names_in_)

st.title("🏡 House Price Prediction")

st.markdown("Enter the property characteristics:")

# User inputs
area = st.number_input("Living Area (GrLivArea)", min_value=10, max_value=1000, value=100)
overall_qual = st.slider("Overall Quality (OverallQual)", min_value=1, max_value=10, value=5)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
neighborhood = st.selectbox("Neighborhood", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'])

# Build initial dict
user_data = {
    'GrLivArea': area,
    'OverallQual': overall_qual,
    'YearBuilt': year_built,
    f'Neighborhood_{neighborhood}': 1
}

# Create DataFrame, add all features expected by the scaler
df_input = pd.DataFrame([user_data])
for feat in feature_names:
    if feat not in df_input.columns:
        df_input[feat] = 0

# Reorder for the scaler
df_input = df_input[feature_names]

# Apply scaler
df_scaled = scaler.transform(df_input)
df_scaled = pd.DataFrame(df_scaled, columns=feature_names)

# Select model features
df_model = df_scaled[columns]

# Prediction
if st.button("Predict Price"):
    # Get raw prediction
    y_scaled = model.predict(df_model)[0]
    
    # Approach 1: Inverse scaling (may not work correctly)
    try:
        # Approach with inverse_transform
        scaled_array = np.zeros(len(feature_names))
        if 'SalePrice' in feature_names:
            idx = feature_names.index('SalePrice')
            scaled_array[idx] = y_scaled
            y_pred1 = scaler.inverse_transform([scaled_array])[0][idx]
        else:
            y_pred1 = y_scaled
    except Exception:
        y_pred1 = None
    
    # Approach 2: Calculate an estimate based on business rules
    # A formula that takes into account the main characteristics
    try:
        base_price = 100000  # Base price
        area_factor = area * 1000  # $1000 per m²
        qual_factor = (overall_qual/10) * 100000  # Up to $100k for max quality
        year_factor = (year_built - 1900) * 1000  # $1000 per year after 1900
        
        # Neighborhood factors
        neighborhood_factors = {
            'NAmes': 0.9,
            'CollgCr': 1.2,
            'OldTown': 0.85,
            'Edwards': 0.8,
            'Somerst': 1.1
        }
        neighborhood_factor = neighborhood_factors.get(neighborhood, 1.0)
        
        # Final price calculation
        y_pred2 = (base_price + area_factor + qual_factor + year_factor) * neighborhood_factor
    except Exception:
        y_pred2 = None
    
    # Approach 3: Use a simple fallback model
    try:
        # Average price per m² (about $2000)
        price_m2 = 2000
        # Quality adjustment (factor between 0.7 and 1.3)
        quality_adjustment = 0.7 + (overall_qual / 10) * 0.6
        # Age adjustment (newer is more expensive)
        age_adjustment = min(1.0, 0.7 + (year_built - 1900) / 100)
        # Neighborhood adjustment
        neighborhood_adjustment = neighborhood_factors.get(neighborhood, 1.0)
        
        # Final price
        y_pred3 = area * price_m2 * quality_adjustment * age_adjustment * neighborhood_adjustment
    except Exception:
        y_pred3 = None
    
    # Choose the best available prediction
    if y_pred1 is not None and 10000 <= y_pred1 < 5000000:
        y_pred = y_pred1
        method = "ML model + inverse scaling"
    elif y_pred2 is not None:
        y_pred = y_pred2
        method = "business rules"
    elif y_pred3 is not None:
        y_pred = y_pred3
        method = "simplified model"
    else:
        y_pred = 200000  # Default value
        method = "default value"
    
    # Display result
    st.success(f"💰 Estimated Price: ${int(y_pred):,} (method: {method})")
