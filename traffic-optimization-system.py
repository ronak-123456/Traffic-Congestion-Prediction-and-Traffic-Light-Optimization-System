import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from scipy.optimize import minimize
import time
import random

# Simulated data generation
def generate_traffic_data(num_samples):
    time_of_day = np.random.randint(0, 24, num_samples)
    day_of_week = np.random.randint(0, 7, num_samples)
    weather = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], num_samples)
    special_events = np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
    
    traffic_volume = (
        10 * np.sin(time_of_day * np.pi / 12) +
        5 * np.cos(day_of_week * np.pi / 3.5) +
        np.random.normal(0, 2, num_samples) +
        np.where(weather == 'Rainy', 5, 0) +
        np.where(special_events == 1, 10, 0)
    )
    
    traffic_volume = np.maximum(traffic_volume, 0)
    
    return pd.DataFrame({
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'weather': weather,
        'special_events': special_events,
        'traffic_volume': traffic_volume
    })

# Data preprocessing
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['weather'])
    X = df_encoded.drop('traffic_volume', axis=1)
    y = df_encoded['traffic_volume']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Train Random Forest model
def train_rf_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Train Neural Network model
def train_nn_model(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    return model

# Ensemble prediction
def ensemble_predict(rf_model, nn_model, X):
    rf_pred = rf_model.predict(X)
    nn_pred = nn_model.predict(X).flatten()
    return (rf_pred + nn_pred) / 2

# Traffic light optimization
def optimize_traffic_lights(traffic_volumes, num_intersections):
    def objective(x):
        return np.sum(np.abs(traffic_volumes - x))
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 120}]
    bounds = [(10, 60)] * num_intersections
    
    result = minimize(objective, x0=np.full(num_intersections, 120/num_intersections),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

# Main function
def main():
    print("Traffic Congestion Prediction and Traffic Light Optimization System")
    
    # Generate and preprocess data
    data = generate_traffic_data(10000)
    X, y, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("Training models...")
    rf_model = train_rf_model(X_train, y_train)
    nn_model = train_nn_model(X_train, y_train)
    
    # Evaluate models
    y_pred = ensemble_predict(rf_model, nn_model, X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Simulate real-time traffic prediction and optimization
    print("\nSimulating real-time traffic prediction and optimization...")
    num_intersections = 4
    for _ in range(5):
        # Generate current traffic conditions
        current_data = generate_traffic_data(num_intersections)
        X_current, _, _ = preprocess_data(current_data)
        
        # Predict traffic volumes
        predicted_volumes = ensemble_predict(rf_model, nn_model, X_current)
        
        # Optimize traffic light timings
        optimized_timings = optimize_traffic_lights(predicted_volumes, num_intersections)
        
        print("\nCurrent Traffic Conditions:")
        print(current_data)
        print("\nPredicted Traffic Volumes:")
        print(predicted_volumes)
        print("\nOptimized Traffic Light Timings (seconds):")
        print(optimized_timings)
        
        time.sleep(2)

if __name__ == "__main__":
    main()
