# Traffic Congestion Prediction and Traffic Light Optimization System

## Overview

This project implements a machine learning-based system for predicting traffic congestion and optimizing traffic light timings in a city. It combines advanced data processing, ensemble machine learning techniques, and optimization algorithms to provide a comprehensive solution for urban traffic management.

## Features

- **Data Generation**: Simulates realistic traffic data based on various factors such as time of day, day of the week, weather conditions, and special events.
- **Machine Learning Models**: 
  - Random Forest Regressor
  - Neural Network (using TensorFlow/Keras)
  - Ensemble method combining both models for improved predictions
- **Traffic Light Optimization**: Utilizes scipy's optimization algorithm to balance traffic flow across multiple intersections.
- **Real-time Simulation**: Demonstrates the system's capability to adapt to changing traffic conditions.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow 2.x
- SciPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ronak-123456/traffic-optimization-system.git
   cd traffic-optimization-system
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install numpy pandas scikit-learn tensorflow scipy
   ```

## Usage

Run the main script to start the traffic congestion prediction and optimization simulation:

```
python traffic_optimization.py
```

The script will:
1. Generate simulated traffic data
2. Train the machine learning models
3. Evaluate the models' performance
4. Run a simulation of real-time traffic prediction and traffic light optimization

## Project Structure

- `traffic_optimization.py`: Main script containing all the functions and logic
- `README.md`: This file, containing project documentation

## How It Works

1. **Data Generation**: The system creates synthetic traffic data that mimics real-world patterns, including time-based fluctuations and the impact of weather and special events.

2. **Data Preprocessing**: The generated data is encoded and scaled to prepare it for machine learning models.

3. **Model Training**: Two models (Random Forest and Neural Network) are trained on the preprocessed data.

4. **Ensemble Prediction**: The system combines predictions from both models to achieve higher accuracy.

5. **Traffic Light Optimization**: Using the predicted traffic volumes, the system optimizes traffic light timings across multiple intersections to minimize overall congestion.

6. **Real-time Simulation**: The script simulates changing traffic conditions and demonstrates how the system adapts its predictions and optimizations in real-time.

## Future Improvements

- Incorporate real data sources for traffic information
- Improve the optimization algorithm to handle more complex traffic scenarios
- Add a user interface for easier interaction with the system
- Implement a more sophisticated traffic simulation model
- Extend the system to handle larger scale city-wide traffic management

## Contributing

Contributions to improve the system are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
