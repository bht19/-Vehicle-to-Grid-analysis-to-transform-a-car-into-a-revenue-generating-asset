import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib

# dataset
file_path = 'historical-irish-electricity-prices.csv'
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'])

def create_features(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Hour'] = data['Date'].dt.hour
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    return data

data_hourly = data.copy()
data_hourly = create_features(data_hourly)

X_hourly = data_hourly.drop(columns=['Price', 'Date'])
y_hourly = data_hourly['Price']

X_train_hourly, X_test_hourly, y_train_hourly, y_test_hourly = train_test_split(X_hourly, y_hourly, test_size=0.2, random_state=42)

# Random Forest Regressor for hourly timeframe
rf_model_hourly = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

# Train the model
rf_model_hourly.fit(X_train_hourly, y_train_hourly)

# Make predictions for the next hour
y_pred_hourly = rf_model_hourly.predict(X_test_hourly)

# Evaluate the model
mse_hourly = mean_squared_error(y_test_hourly, y_pred_hourly)
mae_hourly = mean_absolute_error(y_test_hourly, y_pred_hourly)
rmse_hourly = np.sqrt(mse_hourly)

# actual vs. predicted for hourly predictions
plt.figure(figsize=(14, 7))
plt.plot(y_test_hourly.values, label='Actual')
plt.plot(y_pred_hourly, label='Predicted')
plt.title('Actual vs Predicted Electricity Prices (Hourly)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

data_daily = data.set_index('Date').resample('D').mean().reset_index()
data_daily = create_features(data_daily)

X_daily = data_daily.drop(columns=['Price', 'Date'])
y_daily = data_daily['Price']

X_train_daily, X_test_daily, y_train_daily, y_test_daily = train_test_split(X_daily, y_daily, test_size=0.2, random_state=42)

# Random Forest Regressor for next day
rf_model_daily = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

# Train the model
rf_model_daily.fit(X_train_daily, y_train_daily)

# Make predictions for the next day
y_pred_daily = rf_model_daily.predict(X_test_daily)

# Evaluate the model
mse_daily = mean_squared_error(y_test_daily, y_pred_daily)
mae_daily = mean_absolute_error(y_test_daily, y_pred_daily)
rmse_daily = np.sqrt(mse_daily)

# Plot actual vs. predicted for daily predictions
plt.figure(figsize=(14, 7))
plt.plot(y_test_daily.values, label='Actual')
plt.plot(y_pred_daily, label='Predicted')
plt.title('Actual vs Predicted Electricity Prices (Daily)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Model performance metrics
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'RMSE'],
    'Hourly': [mse_hourly, mae_hourly, rmse_hourly]
})

# Save the models 
joblib.dump(rf_model_hourly, 'rf_model_hourly.pkl')
joblib.dump(rf_model_daily, 'rf_model_daily.pkl')