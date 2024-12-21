import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example data for 2022 and 2023 (hypothetical data)
data = {
    'Year': [2022, 2022, 2022, 2023, 2023, 2023],
    'Month': ['March', 'April', 'May', 'March', 'April', 'May'],
    'Temperature': [28, 30, 35, 30, 32, 38],  # Temperature in °C
    'Humidity': [60, 55, 50, 62, 58, 53],  # Humidity in %
    'Prev_Power_Consumption': [500, 510, 520, 530, 540, 550],  # kWh
    'Power_Consumption': [510, 520, 530, 540, 550, 560]  # kWh (target variable)
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare the feature set (X) and target variable (y)
X = df[['Temperature', 'Humidity', 'Prev_Power_Consumption']]
y = df['Power_Consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting model (XGBoost)
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Predict power consumption on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict the power consumption for 2024 (March, April, May) based on the given features
data_2024 = {
    'Month': ['March', 'April', 'May'],
    'Temperature': [31, 33, 37],  # Predicted temperatures for 2024
    'Humidity': [60, 58, 55],  # Predicted humidity for 2024
    'Prev_Power_Consumption': [560, 570, 580]  # Estimated previous consumption from 2023
}

df_2024 = pd.DataFrame(data_2024)

# Make predictions for 2024
X_2024 = df_2024[['Temperature', 'Humidity', 'Prev_Power_Consumption']]
predictions_2024 = model.predict(X_2024)

# Add predictions to the DataFrame for 2024
df_2024['Predicted_Power_Consumption'] = predictions_2024

# Plotting the results for each year (2022, 2023, 2024)
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Plot for 2022
axs[0].plot(df['Month'][:3], df['Power_Consumption'][:3], label='Actual Power Consumption (2022)', marker='o', color='blue')
axs[0].set_title('Actual Power Consumption for 2022', fontsize=14)
axs[0].set_xlabel('Month', fontsize=12)
axs[0].set_ylabel('Power Consumption (kWh)', fontsize=12)
axs[0].grid(True)

# Plot for 2023
axs[1].plot(df['Month'][3:], df['Power_Consumption'][3:], label='Actual Power Consumption (2023)', marker='o', color='green')
axs[1].set_title('Actual Power Consumption for 2023', fontsize=14)
axs[1].set_xlabel('Month', fontsize=12)
axs[1].set_ylabel('Power Consumption (kWh)', fontsize=12)
axs[1].grid(True)

# Plot for 2024
axs[2].plot(df_2024['Month'], df_2024['Predicted_Power_Consumption'], label='Predicted Power Consumption (2024)', marker='o', color='red')
axs[2].set_title('Predicted Power Consumption for 2024', fontsize=14)
axs[2].set_xlabel('Month', fontsize=12)
axs[2].set_ylabel('Power Consumption (kWh)', fontsize=12)
axs[2].grid(True)

# Adjust layout
plt.tight_layout()

# Show all the plots
plt.show()

# Display predicted power consumption for 2024
print("\nPredicted Power Consumption for 2024 (March, April, May):")
print(df_2024[['Month', 'Predicted_Power_Consumption']])

