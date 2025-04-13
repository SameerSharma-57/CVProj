# A Kalman Filter can help smooth out the tracking of marker corners between frames, making the video stable and smoother, especially "When the markers are temporarily occluded or poorly detected or maybe noisy due to camera shake or other factors."

# A Kalman Filter combines predictions (from previous states) with (Current / New) Measurements

# Handles Noisy Measurements

# Continues to predict when measurements are "temporarily unavailable"

import numpy as np
import cv2
from typing import Optional, Tuple
from tracking import TrackCorners

class MarkerKalmanFilter:
    """
    Kalman filter implementation for tracking ArUco marker corners.
    """
    def __init__(self, initial_corners=None):
        """
        Initialize the Kalman filter for tracking 4 corners (8 state variables: x,y for each corner).
        
        Args:
            initial_corners: Initial corners position (4x2 numpy array)
        """
        # State: [x1, y1, x2, y2, x3, y3, x4, y4, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4]
        # Each corner has position (x,y) and velocity (vx,vy)
        self.kalman = cv2.KalmanFilter(16, 8)
        
        # State transition matrix (A)
        # Position = previous_position + velocity
        # Velocity = velocity (constant velocity model)
        self.kalman.transitionMatrix = np.eye(16, dtype=np.float32)
        for i in range(8):
            self.kalman.transitionMatrix[i, i+8] = 1.0  # Add velocity to position
        
        # Measurement matrix (H)
        # We can only measure the positions, not velocities
        self.kalman.measurementMatrix = np.zeros((8, 16), dtype=np.float32)
        for i in range(8):
            self.kalman.measurementMatrix[i, i] = 1.0
        
        # Process noise covariance matrix (Q)
        # How much we trust our motion model
        #self.kalman.processNoiseCov = np.eye(16, dtype=np.float32) * 0.03
        self.kalman.processNoiseCov = np.eye(16, dtype=np.float32) * 0.008
        
        # Measurement noise covariance matrix (R)
        # How much we trust our measurements
        #self.kalman.measurementNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        self.kalman.measurementNoiseCov = np.eye(8, dtype=np.float32) * 0.25
        
        # Error covariance matrix (P)
        # Initial uncertainty
        self.kalman.errorCovPost = np.eye(16, dtype=np.float32) * 1.0
        
        # Initialize with given corners if provided
        if initial_corners is not None:
            self.initialize(initial_corners)
        
        self.initialized = False
    
    def initialize(self, corners):
        """
        Initialize the Kalman filter with the first detected corners.
        
        Args:
            corners: Array of shape (4,2) representing the 4 corners of the marker
        """
        # Flatten corners to [x1, y1, x2, y2, x3, y3, x4, y4]
        corners_flat = corners.reshape(-1)
        
        # Set initial state
        state = np.zeros(16, dtype=np.float32)
        state[:8] = corners_flat  # Positions
        state[8:] = 0  # Initial velocities = 0
        self.kalman.statePost = state.reshape(-1, 1)
        
        self.initialized = True
    
    def predict(self):
        """
        Predict the next state based on current state and motion model.
        
        Returns:
            Predicted corners as 4x2 numpy array
        """
        state = self.kalman.predict()
        predicted_corners = state[:8].reshape(4, 2)
        return predicted_corners
    
    def correct(self, measured_corners):
        """
        Correct the prediction using the measured corners.
        
        Args:
            measured_corners: Measured corners (4x2 numpy array)
            
        Returns:
            Corrected corners as 4x2 numpy array
        """
        # Flatten corners to [x1, y1, x2, y2, x3, y3, x4, y4]
        measured_corners_flat = measured_corners.reshape(-1, 1).astype(np.float32)
        
        # Perform correction
        state = self.kalman.correct(measured_corners_flat)
        corrected_corners = state[:8].reshape(4, 2)
        return corrected_corners

def KalmanCorners(frame, prev_frame, prev_corners, marker_dict=None, params=None) -> Tuple[np.ndarray, bool]:
    """
    Combined function to track marker corners using Kalman filter, detection and optical flow.
    
    Args:
        frame: Current video frame
        prev_frame: Previous video frame
        prev_corners: Previously detected/tracked corners (4x2 numpy array)
        marker_dict: ArUco marker dictionary (optional, only needed for detection)
        params: ArUco detection parameters
        
    Returns:
        corners: Updated corners (4x2 numpy array)
        is_detected: Boolean flag indicating if corners were detected in this frame
    """
    # Static variable to store the Kalman filter instance
    if not hasattr(KalmanCorners, "kalman_filter"):
        KalmanCorners.kalman_filter = MarkerKalmanFilter()
    
    # Initialize Kalman filter if not done yet
    if not KalmanCorners.kalman_filter.initialized and prev_corners is not None:
        KalmanCorners.kalman_filter.initialize(prev_corners)
    
    is_detected = False
    detected_corners = None
    
    # Try to detect markers
    if frame is not None and marker_dict is not None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, marker_dict, parameters=params)
        
        if ids is not None and len(ids) > 0:
            detected_corners = corners[0][0]  # Use first detected marker's corners
            is_detected = True
    
    # Calculate optical flow tracking (if we have previous data)
    tracked_corners = None
    if prev_frame is not None and prev_corners is not None and frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert corners to format expected by calcOpticalFlowPyrLK
        # print("Prev points shape: ", prev_corners.shape)
        # prev_points = prev_corners.reshape(-1, 1, 2).astype(np.float32)
        
        # Calculate optical flow using Lucas-Kanade method
        tracked_corners = TrackCorners(prev_corners, prev_gray, gray)
        if tracked_corners.shape != (4, 2):
            tracked_corners = None
    
        # Calculate optical flow
        # tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
        #     prev_gray, gray, prev_points, None,
        #     winSize=(15, 15),
        #     maxLevel=2,
        #     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        # )
        
        # # Check if tracking was successful for all points
        # if np.all(status == 1):
        #     tracked_corners = tracked_points.reshape(4, 2)
    
    # Get Kalman prediction
    kalman_prediction = KalmanCorners.kalman_filter.predict()
    
    # Choose the best source for updating
    if is_detected:
        # Use detection and update Kalman filter
        final_corners = KalmanCorners.kalman_filter.correct(detected_corners)
    elif tracked_corners is not None:
        # Use optical flow and update Kalman filter
        final_corners = KalmanCorners.kalman_filter.correct(tracked_corners)
    else:
        # Use only prediction if no other sources available
        final_corners = kalman_prediction
    
    return final_corners, is_detected

def DetectArucoCorners(image, marker_dict=None, parameters=None):
    """
    Detect ArUco markers in the image.
    
    Args:
        image: Input image
        marker_dict: ArUco marker dictionary
        parameters: ArUco detection parameters
        
    Returns:
        corners: List of corner arrays or None
        ids: List of marker IDs or None
    """
    if marker_dict is None:
        marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()
    
    frame = image.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, marker_dict, parameters=parameters)
    
    if ids is not None:
        return corners, ids
    else:
        return None, None
    
    