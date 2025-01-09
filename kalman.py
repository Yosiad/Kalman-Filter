import numpy as np

class KalmanFilter:
    def __init__(self, A, H, Q, R, B=None, u=None):
        """
        Kalman Filter Initialization.
        
        Args:
        - A: State transition matrix (nxn)
        - H: Observation matrix (mxn)
        - Q: Process noise covariance (nxn)
        - R: Measurement noise covariance (mxm)
        - B: Control matrix (nxp), optional
        - u: Control vector (px1), optional
        """
        self.A = A    # State transition matrix
        self.H = H    # Observation matrix
        self.Q = Q    # Process noise covariance 
        self.R = R    # Measurement noise covariance
        self.B = B    # Control matrix
        self.u = u    # Control vector
        self.x = None # Initial state estimate
        self.P = None # Initial error covariance

    def initialize(self, x0, P0):
        """
        Initialize the state and covariance matrix.
        
        Args:
        - x0: Initial state estimate (nx1)
        - P0: Initial error covariance matrix (nxn)
        """
        self.x = x0
        self.P = P0

    def predict(self):
        """
        Predict the next state and covariance.
        """
        # Predict state
        self.x = np.dot(self.A, self.x)
        if self.B is not None and self.u is not None:
            self.x += np.dot(self.B, self.u)
        
        # Predict error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        Update the state estimate and covariance using the measurement.
        
        Args:
        - z: Observation/Measurement (mx1)
        """ 
        # Compute Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update estimate with measurement
        y = z - np.dot(self.H, self.x)  # Measurement residual
        self.x = self.x + np.dot(K, y)
        
        # Update error covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self):
        """
        Return the current state estimate.
        """
        return self.x





# System Model
dt = 1  # Time step
A = np.array([[1, dt],    # State transition matrix
              [0,  1]])  
H = np.array([[1, 0]])    # Observation matrix
Q = np.array([[1e-4, 0],  # Process noise covariance
              [0, 1e-4]])
R = np.array([[1]])       # Measurement noise covariance
B = None                  # No control input
u = None

# Initialize the filter
kf = KalmanFilter(A, H, Q, R)
x0 = np.array([[0],       # Initial position
               [1]])      # Initial velocity
P0 = np.eye(2)            # Initial error covariance
kf.initialize(x0, P0)

# Simulated measurements (noisy position data)
measurements = [1, 2, 3.1, 4.05, 5.2, 6.0, 7.1]

# Apply Kalman Filter
estimated_states = []
for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))
    estimated_states.append(kf.get_state().flatten())

# Print results
for i, state in enumerate(estimated_states):
    print(f"Time {i+1}: Position={state[0]:.2f}, Velocity={state[1]:.2f}")
