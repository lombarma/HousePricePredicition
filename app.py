import streamlit as st
import pandas as pd
import numpy as np

st.title("🏡 House Price Prediction")

st.markdown("Enter the property characteristics:")

# User inputs
area = st.number_input("Living Area (GrLivArea)", min_value=10, max_value=1000, value=100)
overall_qual = st.slider("Overall Quality (OverallQual)", min_value=1, max_value=10, value=5)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
neighborhood = st.selectbox("Neighborhood", ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'])

# Prediction
if st.button("Predict Price"):
    # Business rule approach
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
        y_pred = (base_price + area_factor + qual_factor + year_factor) * neighborhood_factor
        method = "business rules"
    except Exception:
        # Fallback to simple model if business rule calculation fails
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
            y_pred = area * price_m2 * quality_adjustment * age_adjustment * neighborhood_adjustment
            method = "simplified model"
        except Exception:
            # Last resort - default value
            y_pred = 200000
            method = "default value"
    
    # Display result
    st.success(f"💰 Estimated Price: ${int(y_pred):,} (method: {method})")
