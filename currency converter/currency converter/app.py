import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet

# Load the dataset (assuming the dataset is stored in a CSV file)
data = pd.read_csv('bike_rental_data.csv')

# Preprocess the data (e.g., handle missing values, feature engineering)
# Here, you would preprocess the data according to your specific dataset

# Split the data into features and target variable
X = data[['feature1', 'feature2', ...]]  # Features such as time of day, weather conditions, etc.
y = data['rental_count']  # Target variable: bike rental count

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("Random Forest MAE:", mae_rf)

# Train a Prophet model
prophet_model = Prophet()
prophet_data = pd.DataFrame()
prophet_data['ds'] = data['timestamp']  # Assuming 'timestamp' column contains datetime values
prophet_data['y'] = data['rental_count']
prophet_model.fit(prophet_data)

# Make future predictions with Prophet
future = prophet_model.make_future_dataframe(periods=30)  # Predicting for the next 30 days
forecast = prophet_model.predict(future)

# Visualize the forecast
prophet_model.plot(forecast)

# Display the forecast components
prophet_model.plot_components(forecast)
