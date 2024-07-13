# -Vehicle-to-Grid-analysis-to-transform-a-car-into-a-revenue-generating-asset
This project investigates the capability of Vehicle-to-Grid (V2G) technology to convert electric vehicles (EVs) into sources of revenue. The main objective is to create a machine learning model that can forecast electricity prices and develop a web application that provides recommendations for the best charging times.Utilizing V2G technology, EV owners can lower their charging expenses and potentially generate income by feeding energy back into the grid during peak demand periods.

# Project Overview
With the increasing adoption of electric vehicles, optimizing the charging time to minimize electricity costs has become a significant concern. This project aims to address this issue by developing a comprehensive system that predicts electricity prices and identifies the most cost-effective times for EV owners to charge their vehicles. The integration of V2G technology allows EVs to become dynamic assets, capable of both consuming and supplying electricity based on real-time price fluctuations and grid demand.

# Key Features
Electricity Price Prediction: The core of the project is a machine learning model, specifically a Random Forest Regressor, designed to predict future electricity prices using historical data. This model analyzes patterns and trends to provide accurate forecasts, enabling better decision-making for EV owners.

Optimal Charging Time Recommendation: The web application utilizes the predictions from the machine learning model to suggest optimal charging times. By charging during low-cost periods and potentially discharging during high-demand periods, users can save money and earn revenue.

Web Application: Built with the MERN stack (MongoDB, Express.js, React, Node.js), the web application offers a user-friendly interface where EV owners can input their charging preferences. The application then provides real-time recommendations based on the latest electricity price predictions.

Data Visualization: The application also includes data visualization features, allowing users to see the actual vs. predicted electricity prices, historical trends, and potential savings or earnings. This transparency helps users understand the benefits of optimal charging strategies and V2G technology.

# To run the website in your local environment:

Step-1 : Either download the project file or clone the repository

Step-2 : Make sure to keep the files inside the folder in the below order,

Project
|
|
|----Static
|      |------ style.css
|      |------ ev-background.jpg
|      |------ loading.svg      
|
|
|----Templates
|      |------ index.html
|
|---- app.py
|---- model.py
|---- historical-irish-electricity-prices.csv
|---- rf_model_daily.pkl
|---- rf_model_hourly.pkl

Step-3 : Open the terminal and redirect to the project folder. 

Step-4 : Run the code "Python app.py" in the terminal and click on the link generated in the terminal to redirect to the website.

# Requirements:
Flask==2.0.2
Flask-Cors==3.0.10
pandas==1.3.3
numpy==1.21.2
sklearn==0.0
python-dotenv==0.19.1
Python 3.8 or Higher: Python 3.8, 3.9, or 3.10 are generally recommended
