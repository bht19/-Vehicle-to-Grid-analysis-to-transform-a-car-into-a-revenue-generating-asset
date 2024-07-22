from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained models
rf_model_hourly = joblib.load('rf_model_hourly.pkl')

# Load the data and convert prices from megawatts to kilowatts
data = pd.read_csv('historical-irish-electricity-prices.csv')
data['Price_kWh'] = data['Price'] / 1000  # Assuming the column name is 'Price'

def create_features(date):
    data = pd.DataFrame({'Date': [date]})
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Hour'] = data['Date'].dt.hour
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    return data.drop(columns=['Date'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    hours = int(request.form['hours'])
    battery_capacity = float(request.form['battery_capacity'])
    initial_soc = float(request.form['soc'])
    
    start_date = datetime.strptime(date, '%Y-%m-%dT%H:%M')
    input_day_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    predictions = []
    for i in range(24):
        future_date = input_day_start + timedelta(hours=i)
        features = create_features(future_date)
        predicted_price_mwh = rf_model_hourly.predict(features)[0]
        predicted_price_kwh = predicted_price_mwh / 1000  # Convert from MWh to kWh
        predictions.append((future_date, predicted_price_kwh))
    
    predictions.sort(key=lambda x: x[1])
    best_time = predictions[0][0]
    best_price = predictions[0][1]
    
    user_features = create_features(start_date)
    user_price_mwh = rf_model_hourly.predict(user_features)[0]
    user_price_kwh = user_price_mwh / 1000  # Convert from MWh to kWh
    
    amount_charged_at_input_time = user_price_kwh * hours
    amount_charged_at_best_time = best_price * hours
    amount_saved = amount_charged_at_input_time - amount_charged_at_best_time
    
    # Calculate final SOC based on the charger rate of 7 kWh per hour
    energy_added = 7 * hours  # Energy added in kWh
    energy_added_percentage = (energy_added / battery_capacity) * 100
    final_soc = initial_soc + energy_added_percentage
    final_soc = min(final_soc, 100)  # Ensure SOC doesn't exceed 100%

    response = {
        'best_time': best_time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_price': best_price,
        'user_price': user_price_kwh,
        'amount_charged_at_input_time': amount_charged_at_input_time,
        'amount_charged_at_best_time': amount_charged_at_best_time,
        'amount_saved': amount_saved,
        'initial_soc': initial_soc,
        'final_soc': final_soc,
        'predictions': [{'time': p[0].strftime('%Y-%m-%d %H:%M:%S'), 'price': p[1]} for p in predictions]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
