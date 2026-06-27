import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the trained model, scaler, and feature columns
@st.cache_resource
def load_model_components():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model_components()

st.title('Travel Cost Prediction App')

st.write("Enter the details to predict the total trip cost.")

# User inputs (example fields - adapt to your X features)
age_range = st.sidebar.slider('Age Range', 16, 60, 25)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
destination = st.sidebar.selectbox('Destination', ['Jaflong', 'Madhabkunda Lake', 'Ratargul Swamp Forest', 'Utma Chora', 'Bichanakandi', 'Lalakhal', 'Paharpur Buddhist Vihara', 'Cox Bazar'])
transport_mode = st.sidebar.selectbox('Transport Mode', ['Bike', 'Bus', 'CNG / Auto', 'Leguna', 'Private Car', 'Bus + CNG', 'Train'])
main_travel_cost_bdt = st.sidebar.number_input('Main Travel Cost (BDT)', 100, 5000, 500)
stayed_overnight = st.sidebar.selectbox('Stayed Overnight', ['Yes', 'No'])
hotel_cost_per_night_bdt = st.sidebar.number_input('Hotel Cost Per Night (BDT)', 0, 6000, 1000)
food_cost_per_day_bdt = st.sidebar.number_input('Food Cost Per Day (BDT)', 50, 5500, 300)
local_transport_cost_per_day_bdt = st.sidebar.number_input('Local Transport Cost Per Day (BDT)', 0, 5000, 200)
number_of_trip_days = st.sidebar.slider('Number of Trip Days', 1, 40, 3)
number_of_travellers = st.sidebar.slider('Number of Travellers', 1, 40, 5)
travel_season = st.sidebar.selectbox('Travel Season', ['Winter', 'Monsoon', 'Summer'])

# Re-create the engineered features from user input
user_input_df = pd.DataFrame({
    'age_range': [age_range],
    'gender': [gender],
    'destination': [destination],
    'transport_mode': [transport_mode],
    'main_travel_cost_bdt': [main_travel_cost_bdt],
    'stayed_overnight': [stayed_overnight],
    'hotel_cost_per_night_bdt': [hotel_cost_per_night_bdt],
    'food_cost_per_day_bdt': [food_cost_per_day_bdt],
    'local_transport_cost_per_day_bdt': [local_transport_cost_per_day_bdt],
    'number_of_trip_days': [number_of_trip_days],
    'number_of_travellers': [number_of_travellers],
    'travel_season': [travel_season]
})

# Apply feature engineering functions
user_input_df['cost_per_traveller'] = user_input_df['main_travel_cost_bdt'] / user_input_df['number_of_travellers']
user_input_df['avg_daily_expenses_per_person'] = (user_input_df['food_cost_per_day_bdt'] + user_input_df['local_transport_cost_per_day_bdt'] + user_input_df['hotel_cost_per_night_bdt']) / user_input_df['number_of_travellers']
user_input_df['has_hotel'] = user_input_df['stayed_overnight'].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode categorical features
categorical_cols = user_input_df.select_dtypes(include='object').columns
user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_cols, drop_first=True)

# Ensure all feature columns from training are present, fill missing with 0
final_input = pd.DataFrame(columns=feature_columns)
final_input = pd.concat([final_input, user_input_encoded], ignore_index=True)
final_input = final_input.fillna(0)

# Align columns - crucial step to handle cases where some dummy variables might be missing from user input
missing_cols = set(feature_columns) - set(final_input.columns)
for c in missing_cols:
    final_input[c] = 0
final_input = final_input[feature_columns] # Ensure the order of feature columns is the same as in training data


# Scale numerical features
input_scaled = scaler.transform(final_input)

if st.button('Predict Total Trip Cost'):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Total Trip Cost: {prediction:.2f} BDT")

# You can also add sections for the travel suggestion functions
st.subheader("Travel Suggestions")
st.write("Use the functions `suggest_travel_options` and `get_combined_travel_suggestions` to provide travel advice.")
st.info("Note: To use the travel suggestion functions, you would need to integrate the logic and dataframes (`df1`, `df3`) into this Streamlit app, or call an API if they are deployed separately.")
