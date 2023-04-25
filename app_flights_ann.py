import pandas as pd
import streamlit as st
import keras
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from keras.models import load_model
import joblib
from keras.utils import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import pydeck as pdk
from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import plotly.graph_objs as go
import time


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Function to get the session state
def get_session_state(**kwargs):
    session_id = st.session_state.get("_session_id")
    if session_id is None:
        import uuid
        session_id = str(uuid.uuid4())
        st.session_state._session_id = session_id

    return st.session_state.get(session_id, _SessionState(**kwargs))

# The SessionState class
class _SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        delattr(self, name)

session_state = get_session_state(source_dest_map=None, source_city=None, destination_city=None)

## set page configuration
st.set_page_config(page_title='Flight Ticket Prediction', page_icon='🛫', layout='wide', initial_sidebar_state='expanded')

##loading the ann model and label encoder
model = load_model("flight_model")
label_encoders = joblib.load("label_encoders.pkl")

# Load the label encoders dictionary from the pickle file
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

## load the copy of the dataset
df = pd.read_csv("flights.csv")

## add page title and content
st.title('Flight Ticket Prediction')
st.write('Please enter set of information below to predict the possible fligt ticket cost: ')

## add image
image = Image.open("booking-flight-ticket.jpg")
st.image(image, use_column_width = True)

# User input
# Define the lists for each category
airlines = ['Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet']
source_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
departure_times = ['Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon', 'Late_Night']
stops = ['one', 'zero', 'two_or_more']
arrival_times = ['Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon', 'Late_Night']
destination_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
classes = ['Economy', 'Business']

# Create columns for each dropdown menu
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8 = st.columns([1, 2])

# Create dropdown menus to select each category
selected_airline = col1.selectbox('Select an airline:', airlines)
selected_source_city = col2.selectbox('Select source city:', source_cities)
selected_departure_time = col5.selectbox('Select departure time:', departure_times)
selected_stops = col4.selectbox('Select number of stops:', stops)
selected_arrival_time = col6.selectbox('Select arrival time:', arrival_times)
selected_destination_city = col3.selectbox('Select destination city:', destination_cities)
selected_class = col7.selectbox('Select class:', classes)

# Add number input fields for duration and days_left
selected_days_left = col8.slider('Enter days left before departure:', min_value=0, max_value=50, value=30, step=1)


# Function to get coordinates for a city
def get_coordinates(city):
    geolocator = Nominatim(user_agent="flight_ticket_prediction")
    location = geolocator.geocode(city)
    return location.latitude, location.longitude

def create_map(source_city, destination_city):
    global session_state
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
    city_coordinates = [get_coordinates(city) for city in cities]
  


    # Create a DataFrame with source and destination cities and their coordinates
    city_data = pd.DataFrame(
        {
            "City": cities,
            "Latitude": [lat for lat, lon in city_coordinates],
            "Longitude": [lon for lat, lon in city_coordinates],
        }
    )

    source_lat, source_lon = get_coordinates(source_city)
    dest_lat, dest_lon = get_coordinates(destination_city)

    session_state.distance = haversine_distance(source_lat, source_lon, dest_lat, dest_lon)

    # Create a map with markers for source and destination cities
    view_state = pdk.ViewState(latitude=source_lat, longitude=source_lon, zoom=5)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_data,
        get_position=["Longitude", "Latitude"],
        get_radius=35000,
        get_fill_color=[255, 140, 50],
        pickable=True,
    )

    # Create the ArcLayer to connect the source and destination cities
    arc_data = pd.DataFrame(
        {
            "from_lat": [source_lat],
            "from_lon": [source_lon],
            "to_lat": [dest_lat],
            "to_lon": [dest_lon]
        }
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_width=1.2,
        get_tilt=19,
        get_source_color=[0,0,255],
        get_target_color=[255,0,0],
        pickable=True,
        auto_highlight=True,
    )

    # Add TextLayer to display city names on the map
    text_layer = pdk.Layer(
        "TextLayer",
        data=city_data,
        get_position=["Longitude", "Latitude"],
        get_text="City",
        get_size=24,
        get_color=[255, 255, 255],
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
        pickable=True,
    )

    # Add both ScatterplotLayer, ArcLayer, and TextLayer to the map
    map = pdk.Deck(layers=[layer, arc_layer, text_layer], initial_view_state=view_state)
    session_state.source_dest_map = map
    st.pydeck_chart(session_state.source_dest_map)

# Display the interactive map
if session_state.source_city != selected_source_city or session_state.destination_city != selected_destination_city:
    session_state.source_city = selected_source_city
    session_state.destination_city = selected_destination_city
    create_map(selected_source_city, selected_destination_city)
else:
    st.pydeck_chart(session_state.source_dest_map)



def encode_inputs(airline, source_city, departure_time, stops, arrival_time, destination_city, fclass, duration, days_left):
    airline_code = label_encoders['airline'].transform([airline])[0]
    source_city_code = label_encoders['source_city'].transform([source_city])[0]
    destination_city_code = label_encoders['destination_city'].transform([destination_city])[0]
    departure_time_code = label_encoders['departure_time'].transform([departure_time])[0]
    arrival_time_code = label_encoders['arrival_time'].transform([arrival_time])[0]
    stops_code = label_encoders['stops'].transform([stops])[0]
    class_code = label_encoders['class'].transform([fclass])[0]
    
    
    ## create input data for prediction
    input_data = np.array([[airline_code, source_city_code, destination_city_code, departure_time_code, arrival_time_code, stops_code,class_code]])
    input_data = pad_sequences(input_data, maxlen=9)
    
    return input_data

# Find the duration from the CSV file based on user input
flights_data = pd.read_csv('flights.csv')
filtered_flights = flights_data[(flights_data['airline'] == selected_airline) &
                                (flights_data['source_city'] == selected_source_city) &
                                (flights_data['departure_time'] == selected_departure_time) &
                                (flights_data['stops'] == selected_stops) &
                                (flights_data['arrival_time'] == selected_arrival_time) &
                                (flights_data['destination_city'] == selected_destination_city)]

if not filtered_flights.empty or st.button('Predict'):
    data = {
        'Metric': ['Duration', 'Distance', 'Actual ticket price', 'Actual ticket price in GBP', 'Predicted ticket price'],
        'Value': ['', '', '', '', ''],
    }
    if not filtered_flights.empty:
        selected_duration = filtered_flights.iloc[0]['duration']
        data['Value'][0] = f"{selected_duration} hours"
        data['Value'][1] = f"{round(session_state.distance, 2)} miles"

        actual_price = filtered_flights.iloc[0]['price']
        actual_price_gbp = actual_price / 100  # Assuming 1 GBP equals 100 INR
        data['Value'][2] = f"{actual_price} INR"
        data['Value'][3] = f"{actual_price_gbp} GBP"

    if st.button('Predict'):
        encoded_input = encode_inputs(selected_airline, selected_source_city, selected_departure_time, selected_stops, selected_arrival_time, selected_destination_city, selected_class, selected_duration, selected_days_left)
        predicted_price = model.predict(encoded_input)[0][0]
        data['Value'][4] = f"{round(predicted_price, 2)} GBP"

    results_df = pd.DataFrame(data)
    st.dataframe(results_df, width=810, height=212)
else:
    st.write('No matching flights found in the data.')



# Add a new function to generate random predictions
def generate_random_predictions(n):
    random_predictions = []

    for _ in range(n):
        # Generate random inputs
        random_airline = random.choice(airlines)
        random_source_city = random.choice(source_cities)
        random_departure_time = random.choice(departure_times)
        random_stops = random.choice(stops)
        random_arrival_time = random.choice(arrival_times)
        random_destination_city = random.choice(destination_cities)
        random_class = random.choice(classes)
        random_days_left = random.randint(0, 50)

        # Find the corresponding flight in the flights_data DataFrame
        matching_flight = flights_data[(flights_data['airline'] == random_airline) &
                                       (flights_data['source_city'] == random_source_city) &
                                       (flights_data['departure_time'] == random_departure_time) &
                                       (flights_data['stops'] == random_stops) &
                                       (flights_data['arrival_time'] == random_arrival_time) &
                                       (flights_data['destination_city'] == random_destination_city)]

        # Get the actual price for the selected flight
        if not matching_flight.empty:
            actual_price = matching_flight.iloc[0]['price']
            actual_price_gbp = actual_price / 100  # Assuming 1 GBP equals 100 INR
        else:
            actual_price_gbp = None

        # Get the encoded input data based on random inputs
        encoded_input = encode_inputs(random_airline, random_source_city, random_departure_time, random_stops,
                                      random_arrival_time, random_destination_city, random_class, selected_duration, random_days_left)

        # Use the model to make
        #  a prediction based on the encoded input data
        predicted_price = model.predict(encoded_input)[0][0]

        # Append the random inputs, actual price, and the predicted price to the list
        random_predictions.append([random_airline, random_source_city, random_departure_time, random_stops,
                                   random_arrival_time, random_destination_city, random_class, random_days_left,
                                   round(actual_price_gbp, 2) if actual_price_gbp else None, round(predicted_price, 2)])

    return random_predictions

num_random_predictions = st.slider('Select the number of random predictions:', min_value=1, max_value=6000, value=3000, step=1)

def calculate_metrics(actual_prices, predicted_prices):
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = math.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    adj_r2 = 1 - (1 - r2) * (len(actual_prices) - 1) / (len(actual_prices) - 2)

    return mae, mse, rmse, r2, mape, adj_r2

import pandas as pd

def display_metrics(metrics):
    mae, mse, rmse, r2, mape, adj_r2 = metrics
    data = {
        'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R2 Score', 'Mean Absolute Percentage Error (MAPE)', 'Adjusted R2 Score'],
        'Value': [mae, mse, rmse, r2, mape, adj_r2]
    }
    metrics_df = pd.DataFrame(data)
    metrics_df['Value'] = metrics_df['Value'].round(4)
    st.dataframe(metrics_df, width=810, height=245)



def plot_metrics(actual_prices, predicted_prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(actual_prices))), y=actual_prices, mode='markers', name='Actual Prices', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=list(range(len(predicted_prices))), y=predicted_prices, mode='markers', name='Predicted Prices'))
    fig.update_layout(title='Actual vs Predicted Prices', xaxis_title='Index', yaxis_title='Price (GBP)')
    fig.add_annotation(x=1, y=1, text=f'Total indices: {len(actual_prices)}', showarrow=False, xanchor='right', yanchor='top')
    st.plotly_chart(fig)

import matplotlib.pyplot as plt

def plot_combined_graph(random_predictions_df):
    fig, ax1 = plt.subplots()

    # Bar plot for number of airlines predicted
    airline_counts = random_predictions_df['Airline'].value_counts()
    ax1.bar(airline_counts.index, airline_counts.values, color='b', alpha=0.6, label='Number of Airlines')
    ax1.set_ylabel('Number of Airlines', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticklabels(airline_counts.index, rotation=90)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

    # Line plot for actual and predicted flight ticket prices
    ax2.plot(random_predictions_df.index, random_predictions_df['Actual Price (GBP)'], color='g', label='Actual Price (GBP)')
    ax2.plot(random_predictions_df.index, random_predictions_df['Predicted Price (GBP)'], color='r', linestyle='--', label='Predicted Price (GBP)')
    ax2.set_ylabel('Flight Ticket Prices (GBP)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    fig.tight_layout()  # Adjust the layout to prevent overlapping labels
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 1), bbox_transform=ax1.transAxes)

    st.pyplot(fig)


if st.button('Generate Random Predictions'):
    start_time = time.time() # Start the timer
    # Generate random predictions and display them in a table
    random_predictions = generate_random_predictions(num_random_predictions)
    random_predictions_df = pd.DataFrame(random_predictions, columns=['Airline', 'Source City', 'Departure Time', 'Stops', 'Arrival Time', 'Destination City', 'Class', 'Days Left', 'Actual Price (GBP)', 'Predicted Price (GBP)'])
    random_predictions_df.dropna(subset=['Actual Price (GBP)'], inplace=True) # Remove rows with NaN values in 'Actual Price (GBP)' column
    st.write(random_predictions_df)

    # Calculate and display the metrics
    actual_prices = random_predictions_df['Actual Price (GBP)'].values
    predicted_prices = random_predictions_df['Predicted Price (GBP)'].values
    
    metrics = calculate_metrics(actual_prices, predicted_prices)
    display_metrics(metrics)

    # Plot the metrics
    plot_metrics(actual_prices, predicted_prices)

    # Plot airline counts as a bar plot in descending order
    airline_counts = random_predictions_df['Airline'].value_counts().sort_values(ascending=False)
    airline_counts_bar = go.Bar(
        x=airline_counts.index,
        y=airline_counts.values,
        text=airline_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title='Number of Airlines Predicted', xaxis=dict(title='Airline'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[airline_counts_bar], layout=layout)
    st.plotly_chart(fig)

    # Plot the source city counts as a bar plot in descending order
    source_city_counts = random_predictions_df['Source City'].value_counts().sort_values(ascending=False)
    source_city_counts_bar = go.Bar(
        x=source_city_counts.index,
        y=source_city_counts.values,
        text=source_city_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Source Cities Predicted', xaxis=dict(title='Source City'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[source_city_counts_bar], layout=layout)
    st.plotly_chart(fig)

    # Plot the destination city counts as a bar plot in descending order
    destination_city_counts = random_predictions_df['Destination City'].value_counts().sort_values(ascending=False)
    destination_city_counts_bar = go.Bar(
        x=destination_city_counts.index,
        y=destination_city_counts.values,
        text=destination_city_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Destination Cities Predicted', xaxis=dict(title='Destination City'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[destination_city_counts_bar], layout=layout)
    st.plotly_chart(fig)


    # Plot the departure time counts as a bar plot in descending order
    departure_time_counts = random_predictions_df['Departure Time'].value_counts().sort_values(ascending=False)
    departure_time_counts_bar = go.Bar(
        x=departure_time_counts.index,
        y=departure_time_counts.values,
        text=departure_time_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Departure Times Predicted', xaxis=dict(title='Departure Time'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[departure_time_counts_bar], layout=layout)
    st.plotly_chart(fig)
    
    # Plot the arrival time counts as a bar plot in descending order
    arrival_time_counts = random_predictions_df['Arrival Time'].value_counts().sort_values(ascending=False)
    arrival_time_counts_bar = go.Bar(
        x=arrival_time_counts.index,
        y=arrival_time_counts.values,
        text=arrival_time_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Arrival Times Predicted', xaxis=dict(title='Arrival Time'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[arrival_time_counts_bar], layout=layout)
    st.plotly_chart(fig)


    # Plot the stops counts as a bar plot in descending order
    stops_counts = random_predictions_df['Stops'].value_counts().sort_values(ascending=False)
    stops_counts_bar = go.Bar(
        x=stops_counts.index,
        y=stops_counts.values,
        text=stops_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Stops Predicted', xaxis=dict(title='Stops'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[stops_counts_bar], layout=layout)
    st.plotly_chart(fig)

    # Plot the class counts as a bar plot in descending order
    class_counts = random_predictions_df['Class'].value_counts().sort_values(ascending=False)
    class_counts_bar = go.Bar(
        x=class_counts.index,
        y=class_counts.values,
        text=class_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Classes Predicted', xaxis=dict(title='Class'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[class_counts_bar], layout=layout)
    st.plotly_chart(fig)

    # Plot the days left counts as a bar plot in descending order
    days_left_counts = random_predictions_df['Days Left'].value_counts().sort_values(ascending=False)
    days_left_counts_bar = go.Bar(
        x=days_left_counts.index,
        y=days_left_counts.values,
        text=days_left_counts.values,
        textposition='auto',
        textfont=dict(color='black',size=22),
    )

    layout = go.Layout(title = 'Number of Days Left Before Flight Predicted', xaxis=dict(title='Days Left'), yaxis=dict(title='Count'))
    fig = go.Figure(data=[days_left_counts_bar], layout=layout)
    st.plotly_chart(fig)
   

    end_time = time.time() # Stop the timer
    st.write(f'Time taken to generate random predictions: {end_time - start_time:.2f} seconds')







