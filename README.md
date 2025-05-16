# House Price Prediction 🏡

A machine learning application that predicts house prices based on property characteristics. Built with Streamlit and scikit-learn.

## Overview

This application allows users to input key property features such as:

- Living area
- Overall quality
- Year built
- Neighborhood

It then uses multiple prediction approaches to estimate the property's value.

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Features

- **Interactive UI**: Easily adjust property parameters and get instant price predictions
- **Multiple Prediction Methods**: Uses a combination of ML model and business rules for robust predictions
- **Real-time Updates**: See how changing property features affects the predicted price

## Model

The application uses a pre-trained machine learning model saved in `model.pkl`. The model is trained on the Ames Housing dataset, which includes a wide variety of housing features.

Three prediction approaches are used:

1. Machine learning model with inverse scaling
2. Business rule-based calculation
3. Simplified model based on property metrics

## Files

- `app.py` - Main Streamlit application
- `model.pkl` - Trained machine learning model
- `columns.pkl` - Feature columns used by the model
- `scaler.pkl` - Data scaler used for feature normalization

## Data

This project uses the Ames Housing dataset, which contains information on residential properties in Ames, Iowa.
