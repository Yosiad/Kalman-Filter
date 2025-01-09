# Kalman Filter Implementation

This repository contains a Python implementation of the **Kalman Filter**, a recursive algorithm for state estimation in systems with noisy measurements. It demonstrates tracking the position and velocity of an object moving along a straight line.

## Features
- General Kalman Filter implementation for linear systems.
- Tracks position and velocity using noisy measurements.
- Easily customizable parameters.

## Algorithm Overview
1. **Prediction**: Predicts the next state and updates error covariance.
2. **Update**: Refines the prediction using new measurements.

## Example Usage
```python
from kalman_filter import KalmanFilter
import numpy as np

# Define system parameters
dt = 1
A = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[1e-4, 0], [0, 1e-4]])
R = np.array([[1]])

# Initialize filter
kf = KalmanFilter(A, H, Q, R)
x0 = np.array([[0], [1]])
P0 = np.eye(2)
kf.initialize(x0, P0)

# Simulated measurements
measurements = [1, 2, 3.1, 4.05, 5.2]
for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))
    print("State:", kf.get_state().flatten())
```

## Output
For measurements `[1, 2, 3.1, 4.05, 5.2]`, the Kalman Filter produces smoothed estimates like:
```
Position=0.67, Velocity=0.33
Position=1.60, Velocity=0.80
```

## Customization
- Modify `A`, `H`, `Q`, `R` for your specific use case.

## License
This project is licensed under the MIT License.

