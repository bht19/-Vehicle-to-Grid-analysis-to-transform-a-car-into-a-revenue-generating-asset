from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained models
rf_model_hourly = joblib.load('rf_model_hourly.pkl')

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
    
    start_date = datetime.strptime(date, '%Y-%m-%dT%H:%M')
    input_day_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    predictions = []
    for i in range(24):
        future_date = input_day_start + timedelta(hours=i)
        features = create_features(future_date)
        predicted_price = rf_model_hourly.predict(features)[0]
        predictions.append((future_date, predicted_price))
    
    predictions.sort(key=lambda x: x[1])
    best_time = predictions[0][0]
    best_price = predictions[0][1]
    
    user_features = create_features(start_date)
    user_price = rf_model_hourly.predict(user_features)[0]
    
    amount_charged_at_input_time = user_price * hours
    amount_charged_at_best_time = best_price * hours
    amount_saved = amount_charged_at_input_time - amount_charged_at_best_time
    
    response = {
        'best_time': best_time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_price': best_price,
        'user_price': user_price,
        'amount_charged_at_input_time': amount_charged_at_input_time,
        'amount_charged_at_best_time': amount_charged_at_best_time,
        'amount_saved': amount_saved,
        'predictions': [{'time': p[0].strftime('%Y-%m-%d %H:%M:%S'), 'price': p[1]} for p in predictions]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
