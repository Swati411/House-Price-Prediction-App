import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# load the model
model = pickle.load(open('linear_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

# title for app
st.title('House Price Prediction App')

# define inputs
sqft = st.number_input('Square Footage', min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
year = st.number_input('Year Built', min_value=1900, max_value=2025, value=2010)
lot = st.number_input('Lot Size', min_value=500, max_value=10000, value=2000)
garage = st.number_input('Garage Size', min_value=0, max_value=5, value=1)
quality = st.number_input('Neighborhood Quality', min_value=1, max_value=10, value=5)

# (No encoding needed because all are numerical)

# create dataframe
input_features = pd.DataFrame({
    'Square_Footage':[sqft],
    'Num_Bedrooms':[bedrooms],
    'Num_Bathrooms':[bathrooms],
    'Year_Built':[year],
    'Lot_Size':[lot],
    'Garage_Size':[garage],
    'Neighborhood_Quality':[quality]
})

# scaling 
scaler = StandardScaler()
input_features = scaler.fit_transform(input_features)

# predictions
if st.button('Predict'):
    predictions = model.predict(input_features)
    output = round(np.exp(predictions[0]),2)
    st.success(f'Predicted Price: ${output}')