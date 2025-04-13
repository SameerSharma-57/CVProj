from calibrate import CalibrateCamera
from marker import EstimateExtrinsicUsingMarkerDetection, EstimateExtrinsicUsingMarkerKalman, EstimateExtrinsicUsingMarkerOpticalFlow
from render import Render
from depth_estimation import DepthFromVideo

'''
Pipeline stages:
1. Calibration (1 method)
2. Finding 2D-3D correspondence (1 method) 
3. Camera extrinsics estimation using correspondences (1 method)
4. Rendering using calculated extrinsics (1 method)
'''

calib_video_path = "data/input/calib_video.mp4"
ar_video_path = "data/input/marker_video.mp4"
output_path = "data/output"
square_size = 19.00 
n_images = 40

print("##"*25)
print("Calibrating camera...")
# CalibrateCamera(calib_video_path, output_path, square_size, n_images)
print("Camera Calibration Done")

# print("##"*25)
# print("Computing depth...")
# DepthFromVideo(ar_video_path, output_path)
# print("Depth Computation Done")

print("##"*25)
print("Estimating extrinsics...")
# EstimateExtrinsicUsingMarker(ar_video_path, output_path)
# KalmanEstimateExtrinsicUsingMarker(ar_video_path, output_path)
EstimateExtrinsicUsingMarkerOpticalFlow(ar_video_path, output_path)
print("Extrinsics Estimation Done")

print("##"*25)
print("Rendering...")
Render(ar_video_path, f"{output_path}", "data/input/cone.obj")
print("Rendering Done")
