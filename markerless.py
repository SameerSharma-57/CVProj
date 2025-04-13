import cv2
import numpy as np
import os

def SfMExtrinsicEstimation(input_video_path, intrinsic_matrix, distortion_coeff, output_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    os.makedirs(os.path.join(output_path, "arrays"), exist_ok=True)

    K = intrinsic_matrix
    dist_coeffs = distortion_coeff
    extrinsics = {}

    # Parameters
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    prev_R = np.eye(3)
    prev_t = np.zeros((3, 1))
    extrinsics['0001'] = np.eye(4)  # First frame is identity

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame from video.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.undistort(prev_gray, K, dist_coeffs)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=2000, qualityLevel=0.01, minDistance=7)

    idx = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.undistort(gray, K, dist_coeffs)

        # Optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        if next_pts is None or status is None:
            break

        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        # Essential matrix
        if len(good_prev) < 8:
            print(f"Frame {idx}: Not enough points. Skipping...")
            continue
        E, mask = cv2.findEssentialMat(good_next, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print(f"Frame {idx}: Essential matrix not found. Skipping...")
            continue

        # Recover relative pose
        _, R, t, mask_pose = cv2.recoverPose(E, good_next, good_prev, K)

        # Compose with previous pose to get current world pose
        curr_t = prev_R @ t + prev_t
        curr_R = prev_R @ R

        # Construct 4x4 extrinsic matrix (world-to-camera)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = curr_R
        extrinsic[:3, 3] = curr_t.flatten()
        key = str(idx).zfill(4)
        extrinsics[key] = extrinsic

        # Update for next iteration
        prev_pts = good_next.reshape(-1, 1, 2)
        prev_gray = gray.copy()
        prev_R, prev_t = curr_R, curr_t
        idx += 1

    cap.release()

    # Save .npz file
    save_path = os.path.join(output_path, "arrays", "extrinsics.npz")
    np.savez(save_path, **extrinsics)
    print(f"Saved {len(extrinsics)} extrinsic matrices to {save_path}")

# command for storing conda env in environment.yml
# conda env export > environment.yml