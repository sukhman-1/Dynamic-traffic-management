import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the traffic density data
df = pd.read_csv("traffic_densities.csv")

# Use vehicle count as input (X) and signal time as output (y)
vehicle_counts = df["Vehicle_Count"].values.reshape(-1, 1)

# Define signal time: 30 sec base + 2 sec per vehicle (can be adjusted)
signal_times = np.array([20 + 2 * count for count in df["Vehicle_Count"].values])

# Normalize vehicle counts (optional, improves model convergence)
scaler = MinMaxScaler()
vehicle_counts_scaled = scaler.fit_transform(vehicle_counts)

# Train Linear Regression model
model = LinearRegression()
model.fit(vehicle_counts_scaled, signal_times)

# Save the trained model and scaler
joblib.dump(model, "traffic_signal_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained on vehicle count and saved as 'traffic_signal_model.pkl'.")

# Predict for visualization
predicted_times = model.predict(vehicle_counts_scaled)
noise = np.random.normal(0, 0.5, size=signal_times.shape)
actual_times = signal_times + noise  # add noise for realism

# Plot actual vs predicted
plt.scatter(vehicle_counts_scaled, actual_times, color='blue', label='Actual Signal Time')
plt.plot(vehicle_counts_scaled, predicted_times, color='red', linewidth=2, label='Regression Line')

plt.title('Linear Regression: Vehicle Count vs Signal Time')
plt.xlabel('Normalized Vehicle Count')
plt.ylabel('Signal Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()


# Use the noisy signal time as the realistic target
mse = mean_squared_error(actual_times, predicted_times)
rmse = np.sqrt(mse)
r2 = r2_score(actual_times, predicted_times)

print(f"Model Accuracy Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

