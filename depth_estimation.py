import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_keypoints_and_descriptors(imgL, imgR):
    """Use ORB detector and FLANN matcher to get keypoints, descritpors,
    and corresponding matches that will be good for computing
    homography.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    ############## Using FLANN matcher ##############
    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    return kp1, des1, kp2, des2, flann_match_pairs


def lowes_ratio_test(matches, ratio_threshold=0.6):
    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue  # skip bad match
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    return filtered_matches

def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matches", img)
    cv2.imwrite("ORB_FLANN_Matches.png", img)
    cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    """Use the set of good mathces to estimate the Fundamental Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for m in matches[:8]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # You can play with the Threshold and confidence values here
        # until you get something that gives you reasonable results. I
        # used the defaults
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            # ransacReprojThreshold=3,
            # confidence=0.99,
        )
    return fundamental_matrix, inliers, pts1, pts2

# def GetDepth(img1, img2, intrinsic_matrix, distortion_coeff):
#     # Step 1: Undistort images
#     img1_undist = cv2.undistort(img1, intrinsic_matrix, distortion_coeff)
#     img2_undist = cv2.undistort(img2, intrinsic_matrix, distortion_coeff)

#     # Step 2: Convert to grayscale
#     gray1 = cv2.cvtColor(img1_undist, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2_undist, cv2.COLOR_BGR2GRAY)

#     # Step 3: StereoSGBM parameters
#     min_disp = 0
#     num_disp = 160  # Must be divisible by 16
#     block_size = 5

#     matcher_params = dict(
#         minDisparity=min_disp,
#         numDisparities=num_disp,
#         blockSize=block_size,
#         P1=8 * 3 * block_size ** 2,
#         P2=32 * 3 * block_size ** 2,
#         disp12MaxDiff=1,
#         uniquenessRatio=10,
#         speckleWindowSize=200,
#         speckleRange=2,
#         preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )

#     # Left and right matchers
#     left_matcher = cv2.StereoSGBM_create(**matcher_params)
#     right_matcher = cv2.StereoSGBM_create(**matcher_params)

#     # Step 4: Compute disparities
#     disp_left = left_matcher.compute(gray1, gray2).astype(np.float32) / 16.0
#     disp_right = right_matcher.compute(gray2, gray1).astype(np.float32) / 16.0

#     # Step 5: WLS filtering for better disparity map from both views
#     try:
#         wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
#         wls_filter.setLambda(8000)
#         wls_filter.setSigmaColor(1.5)
#         disp_left_filtered = wls_filter.filter(disp_left, gray1, disparity_map_right=disp_right)

#         wls_filter_right = cv2.ximgproc.createDisparityWLSFilter(matcher_left=right_matcher)
#         wls_filter_right.setLambda(8000)
#         wls_filter_right.setSigmaColor(1.5)
#         disp_right_filtered = wls_filter_right.filter(disp_right, gray2, disparity_map_right=disp_left)
#     except AttributeError:
#         disp_left_filtered = disp_left
#         disp_right_filtered = disp_right

#     # Step 6: Disparity to depth
#     fx = intrinsic_matrix[0, 0]
#     baseline = 0.1  # Set according to your stereo setup

#     with np.errstate(divide='ignore'):
#         depth1 = fx * baseline / disp_left_filtered
#         depth2 = fx * baseline / disp_right_filtered

#     depth1[disp_left_filtered <= 0] = 0
#     depth2[disp_right_filtered <= 0] = 0

#     # Step 7: Convert to float32 before bilateral filtering
#     depth1 = depth1.astype(np.float32)
#     depth2 = depth2.astype(np.float32)

#     # Step 8: Bilateral filtering
#     depth1 = cv2.bilateralFilter(depth1, d=9, sigmaColor=75, sigmaSpace=75)
#     depth2 = cv2.bilateralFilter(depth2, d=9, sigmaColor=75, sigmaSpace=75)

#     return depth1, depth2

def GetDepth(img1, img2, intrinsic_matrix, distortion_coeff):
    # Step 1: Undistort images
    img1_undist = cv2.undistort(img1, intrinsic_matrix, distortion_coeff)
    img2_undist = cv2.undistort(img2, intrinsic_matrix, distortion_coeff)

    # Step 2: Convert to grayscale
    gray1 = cv2.cvtColor(img1_undist, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_undist, cv2.COLOR_BGR2GRAY)

    # Step 3: StereoSGBM parameters
    min_disp = 0
    num_disp = 128  # Must be divisible by 16
    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Step 4: Compute disparities both ways
    disp_left = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    disp_right = stereo.compute(gray2, gray1).astype(np.float32) / 16.0

    # Step 5: Convert disparity to depth
    fx = intrinsic_matrix[0, 0]
    baseline = 0.1  # You should estimate or provide this

    with np.errstate(divide='ignore'):
        depth_left = fx * baseline / disp_left
        depth_right = fx * baseline / disp_right

    depth_left[disp_left <= 0] = 0
    depth_right[disp_right <= 0] = 0

    return depth_left, depth_right

# def GetDepth(img1, img2, intrinsic_matrix, distortion_coeff):
#     # Step 1: Undistort
#     img1_undist = cv2.undistort(img1, intrinsic_matrix, distortion_coeff)
#     img2_undist = cv2.undistort(img2, intrinsic_matrix, distortion_coeff)

#     # Step 2: Feature matching to estimate baseline and rectify
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(img1_undist, None)
#     kp2, des2 = orb.detectAndCompute(img2_undist, None)
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     matches = bf.knnMatch(des1, des2, k=2)

#     # Lowe's Ratio Test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)

#     if len(good_matches) < 8:
#         return np.zeros_like(img1_undist), np.zeros_like(img2_undist)

#     pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

#     # Step 3: Compute Fundamental and Essential matrices
#     F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
#     if F is None:
#         return np.zeros_like(img1_undist), np.zeros_like(img2_undist)
#     E = intrinsic_matrix.T @ F @ intrinsic_matrix

#     # Recover pose (gives R and t between the two views)
#     _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, intrinsic_matrix)

#     # Estimate baseline (magnitude of t vector)
#     baseline = np.linalg.norm(t)

#     # Step 4: Stereo Rectification (uncalibrated)
#     h, w = img1.shape[:2]
#     _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))

#     img1_rect = cv2.warpPerspective(img1_undist, H1, (w, h))
#     img2_rect = cv2.warpPerspective(img2_undist, H2, (w, h))

#     # Step 5: Stereo Matching (Disparity)
#     min_disp = 0
#     num_disp = 64  # must be divisible by 16
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=min_disp,
#         numDisparities=num_disp,
#         blockSize=9,
#         P1=8 * 3 * 3 ** 2,
#         P2=32 * 3 * 3 ** 2,
#         uniquenessRatio=5,
#         speckleWindowSize=50,
#         speckleRange=2,
#         disp12MaxDiff=1
#     )
#     disp1 = stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16.0
#     disp2 = stereo.compute(img2_rect, img1_rect).astype(np.float32) / 16.0

#     # Step 6: Convert disparity to depth
#     fx = intrinsic_matrix[0, 0]
#     with np.errstate(divide='ignore'):
#         depth1 = fx * baseline / disp1
#         depth2 = fx * baseline / disp2

#     # Step 7: Set invalid regions (disparity <= 0) to large value
#     depth1[disp1 <= 0] = 0
#     depth2[disp2 <= 0] = 0

#     return depth1, depth2

def DepthFromVideo(input_video_path, output_path, d=100):
    os.makedirs(f"{output_path}/arrays", exist_ok=True)
    os.makedirs(f"{output_path}/videos", exist_ok=True)

    intrinsics = np.load(f"{output_path}/arrays/calib_data.npz") # K|0
    intrinsic_matrix = intrinsics["camMatrix"]
    distortion_coefficients = intrinsics["distCoef"]

    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Read all frames into memory
    frames = []
    for _ in tqdm(range(frame_count), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()

    depth_maps = np.zeros((len(frames), H, W), dtype=np.float32)

    for i in tqdm(range(0, len(frames) - d), desc="Computing depth"):
        imgL = frames[i]
        imgR = frames[i + d]

        # Get depth
        depth1, depth2 = GetDepth(imgL, imgR, intrinsic_matrix, distortion_coefficients)
        depth_maps[i][depth_maps[i] == 0] = depth1[depth_maps[i] == 0]
        depth_maps[i+d][depth_maps[i+d] == 0] = depth2[depth_maps[i+d] == 0]
    
    depth_dict = {}
    depth_video_frames = []

    for i in range(len(depth_maps)):
        frame_key = str(i + 1).zfill(4)
        depth_dict[frame_key] = depth_maps[i]
        disp_vis = cv2.normalize(depth_maps[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_video_frames.append(disp_vis)

    # Save depth .npz
    np.savez_compressed(f"{output_path}/arrays/depth.npz", **depth_dict)

    # Save depth video
    out_vid_path = os.path.join(f"{output_path}/videos/depth.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_vid_path, fourcc, fps, (W, H), isColor=False)
    for frame in depth_video_frames:
        out_vid.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))  # convert for 3-channel video
    out_vid.release()

    print(f"Saved depth .npz at {output_path}/depth.npz")
    print(f"Saved depth video at {out_vid_path}")


# def DepthFromVideo(input_video_path, output_path, d=2):
#     os.makedirs(f"{output_path}/arrays", exist_ok=True)
#     os.makedirs(f"{output_path}/videos", exist_ok=True)
    
#     cap = cv2.VideoCapture(input_video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Read all frames into memory
#     frames = []
#     for _ in tqdm(range(frame_count), desc="Reading frames"):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frames.append(gray)
#     cap.release()

#     # Prepare stereo matcher
#     win_size = 5
#     min_disp = 0
#     num_disp = 64  # must be divisible by 16
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=min_disp,
#         numDisparities=num_disp,
#         blockSize=7,
#         uniquenessRatio=5,
#         speckleWindowSize=5,
#         speckleRange=5,
#         disp12MaxDiff=2,
#         P1=8 * 3 * win_size ** 2,
#         P2=32 * 3 * win_size ** 2,
#     )

#     depth_maps = {}
#     depth_video_frames = []

#     for i in tqdm(range(0, len(frames) - d), desc="Computing depth"):
#         imgL = frames[i]
#         imgR = frames[i + d]

#         kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)
#         good_matches = lowes_ratio_test(flann_match_pairs, 0.25)

#         if len(good_matches) < 8:
#             continue

#         F, _, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)
#         if F is None:
#             continue

#         try:
#             _, H1, H2 = cv2.stereoRectifyUncalibrated(
#                 np.float32(points1), np.float32(points2), F, imgSize=(W, H)
#             )
#         except:
#             continue

#         imgL_rect = cv2.warpPerspective(imgL, H1, (W, H))
#         imgR_rect = cv2.warpPerspective(imgR, H2, (W, H))

#         disparity = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0
#         disparity[disparity < 0] = 0  # clean up noise

#         frame_key = str(i + 1).zfill(4)
#         depth_maps[frame_key] = disparity

#         # Normalize disparity for visualization
#         disp_vis = np.uint8(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
#         depth_video_frames.append(disp_vis)

#     # Save depth .npz
#     np.savez_compressed(f"{output_path}/arrays/depth.npz", **depth_maps)

#     # Save depth video
#     out_vid_path = os.path.join(f"{output_path}/videos/depth.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_vid = cv2.VideoWriter(out_vid_path, fourcc, fps, (W, H), isColor=False)
#     for frame in depth_video_frames:
#         out_vid.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))  # convert for 3-channel video
#     out_vid.release()
#     print(f"Saved depth .npz at {output_path}/depth.npz")
#     print(f"Saved depth video at {out_vid_path}")

if __name__ == "__main__":
    imgL, imgR = cv2.imread("data/input/left.jpeg"), cv2.imread("data/input/right.jpeg")
    # imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    # imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    intrinsics = np.load(f"data/output/arrays/calib_data.npz") # K|0
    intrinsic_matrix = intrinsics["camMatrix"]
    distortion_coefficients = intrinsics["distCoef"]

    depth1, depth2 = GetDepth(imgL, imgR, intrinsic_matrix, distortion_coefficients)

    disp_vis1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_vis2 = cv2.normalize(depth2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite("data/output/left_depth.png", disp_vis1)
    cv2.imwrite("data/output/right_depth.png", disp_vis2)

    # ############## Find good keypoints to use ##############
    # kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)
    # good_matches = lowes_ratio_test(flann_match_pairs, 0.2)
    # # draw_matches(imgL, imgR, kp1, des1, kp2, des2, good_matches)


    # ############## Compute Fundamental Matrix ##############
    # F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)


    # ############## Stereo rectify uncalibrated ##############
    # h1, w1 = imgL.shape
    # h2, w2 = imgR.shape
    # thresh = 0
    # _, H1, H2 = cv2.stereoRectifyUncalibrated(
    #     np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=thresh,
    # )

    # ############## Undistort (Rectify) ##############
    # imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
    # imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))
    # cv2.imwrite("undistorted_L.png", imgL_undistorted)
    # cv2.imwrite("undistorted_R.png", imgR_undistorted)

    # ############## Calculate Disparity (Depth Map) ##############

    # # Using StereoBM
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity_BM = stereo.compute(imgL_undistorted, imgR_undistorted)
    # plt.imshow(disparity_BM, "gray")
    # plt.colorbar()
    # plt.show()

    # # Using StereoSGBM
    # # Set disparity parameters. Note: disparity range is tuned according to
    # #  specific parameters obtained through trial and error.
    # win_size = 2
    # min_disp = -6
    # max_disp = 10
    # num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=min_disp,
    #     numDisparities=num_disp,
    #     blockSize=5,
    #     uniquenessRatio=5,
    #     speckleWindowSize=5,
    #     speckleRange=5,
    #     disp12MaxDiff=2,
    #     P1=8 * 3 * win_size ** 2,
    #     P2=32 * 3 * win_size ** 2,
    # )
    # disparity_SGBM = stereo.compute(imgL_undistorted, imgR_undistorted)
    # plt.imshow(disparity_SGBM, "gray")
    # plt.colorbar()
    # plt.show()

