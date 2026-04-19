import streamlit as st
import pandas as pd
import joblib

# Load the saved model pipeline and data
model_pipeline = joblib.load('knn_model_pipeline.joblib')
df_1 = pd.read_csv('df_1.csv')
df_2 = pd.read_csv('df_2.csv')
df_3 = pd.read_csv('df_3.csv')

st.title("Travel Satisfaction Predictor & Fare Guide")

st.sidebar.header("Input Trip Details")

# Collect user inputs matching the features in df_1
age_range = st.sidebar.selectbox("Age Range", options=sorted(df_1['age_range'].unique()))
gender = st.sidebar.selectbox("Gender", options=df_1['gender'].unique())
destination = st.sidebar.selectbox("Destination", options=sorted(df_1['destination'].unique()))
transport_mode = st.sidebar.selectbox("Transport Mode", options=df_1['transport_mode'].unique())
stayed_overnight = st.sidebar.radio("Stayed Overnight?", options=['Yes', 'No'])
travel_season = st.sidebar.selectbox("Travel Season", options=df_1['travel_season'].unique())

main_travel_cost = st.sidebar.number_input("Main Travel Cost (BDT)", min_value=0.0, value=500.0)
hotel_cost = st.sidebar.number_input("Hotel Cost per Night (BDT)", min_value=0.0, value=0.0)
food_cost = st.sidebar.number_input("Food Cost per Day (BDT)", min_value=0.0, value=500.0)
local_transport_cost = st.sidebar.number_input("Local Transport Cost per Day (BDT)", min_value=0.0, value=100.0)
num_days = st.sidebar.number_input("Number of Trip Days", min_value=1, value=1)
num_travellers = st.sidebar.number_input("Number of Travellers", min_value=1, value=1)
total_cost = st.sidebar.number_input("Total Trip Cost (BDT)", min_value=0.0, value=1000.0)

# Prediction Logic
if st.button("Predict Satisfaction"):
    input_data = pd.DataFrame({
        'age_range': [age_range],
        'gender': [gender],
        'destination': [destination],
        'transport_mode': [transport_mode],
        'main_travel_cost_bdt': [main_travel_cost],
        'stayed_overnight': [stayed_overnight],
        'hotel_cost_per_night_bdt': [hotel_cost],
        'food_cost_per_day_bdt': [food_cost],
        'local_transport_cost_per_day_bdt': [local_transport_cost],
        'number_of_trip_days': [num_days],
        'number_of_travellers': [num_travellers],
        'travel_season': [travel_season],
        'total_trip_cost_bdt': [total_cost]
    })
    
    prediction = model_pipeline.predict(input_data)
    # Map numerical prediction back to labels if necessary, 
    # but since we used a Pipeline with LabelEncoder outside or target mapping, 
    # we show the result directly.
    st.subheader(f"Predicted Satisfaction: {prediction[0]}")

# Fare Lookup Logic
st.divider()
st.header("Fare Information Lookup")

fare_type = st.radio("Lookup Type", ["Local Fares (Sylhet Area)", "Bus Routes (Inter-city)"])

if fare_type == "Local Fares (Sylhet Area)":
    search_dest = st.text_input("Search Local Destination (e.g., Jaflong)")
    res = df_3[df_3['Destination'].str.contains(search_dest, case=False, na=False)]
    st.dataframe(res)
else:
    search_route = st.text_input("Search Bus Route (e.g., Barishal)")
    res = df_2[df_2['Route Description'].str.contains(search_route, case=False, na=False)]
    st.dataframe(res)
