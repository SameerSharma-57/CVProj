import streamlit as st
import os
from pathlib import Path
import shutil
import base64
from calibrate import CalibrateCamera
from marker import EstimateExtrinsicUsingMarker
from render import Render

def wait(seconds):
    """Wait for a specified number of seconds."""
    import time
    time.sleep(seconds)

# Set up directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

calib_video_path = "data/input/calib_video.mp4"
ar_video_path = "data/input/marker_video.mp4"
output_path = "data/output"
square_size = 19.00 
n_images = 40

scroll_script = """
<script>
window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
</script>
"""

# Sidebar options
st.sidebar.header("Configuration")
position_method = st.sidebar.selectbox("Positioning Method", ["Marker", "Markerless"])
depth_method = st.sidebar.selectbox("Depth Estimation", ["Stereo", "SfM"])

# Main UI
st.title("Computer Vision Project UI")

# File uploaders
st.subheader("Upload Files")
CALIB_VIDEO_PATH = UPLOAD_DIR / "calibration_video.mp4"
MARKER_VIDEO_PATH = UPLOAD_DIR / "marker_video.mp4"
OBJ_FILE_PATH = UPLOAD_DIR / "model.obj"
calib_video = st.file_uploader("Upload Calibration Video", type=["mp4"], key="calib", label_visibility='collapsed')
if calib_video:
    with open(CALIB_VIDEO_PATH, "wb") as f:
        f.write(calib_video.read())
    st.video(str(CALIB_VIDEO_PATH))


marker_video = st.file_uploader("Upload Marker Video", type=["mp4"], key="marker", label_visibility='collapsed')
if marker_video:
    with open(MARKER_VIDEO_PATH, "wb") as f:
        f.write(marker_video.read())
    st.video(str(MARKER_VIDEO_PATH))


obj_file = st.file_uploader("Upload OBJ File", type=["obj"], key="obj", label_visibility='collapsed')
if obj_file:
    with open(OBJ_FILE_PATH, "wb") as f:
        f.write(obj_file.read())
    st.success("OBJ file uploaded.")


# Save uploaded files



submit = st.button("Submit")

if submit:
    st.subheader("Pipeline Progress")

    # Step 1: Calibration
    with st.spinner("Running Calibration..."):
        try:
            # CalibrateCamera(calib_video_path, output_path, square_size, n_images)
            # wait(2)
            st.success("Calibration Complete ✅")
        except Exception as e:
            st.error(f"Calibration Failed ❌: {e}")

    # Step 2: Estimating Extrinsics
    with st.spinner("Estimating Camera Extrinsics..."):
        try:
            EstimateExtrinsicUsingMarker(ar_video_path, output_path)
            # wait(2)
            st.success("Extrinsics Estimation Complete ✅")
        except Exception as e:
            st.error(f"Extrinsics Estimation Failed ❌: {e}")

    # Step 3: Rendering
    with st.spinner("Rendering Output Video..."):
        try:
            Render(ar_video_path, f"{output_path}", "data/input/cone.obj")
            # wait(2)
            output_video_path = "data/output/videos/3_render.mp4"
            if os.path.exists(output_video_path):
                st.success("Rendering Complete ✅")
                st.video(str(output_video_path))
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                    b64 = base64.b64encode(video_bytes).decode()
                    href = f'<a href="data:video/mp4;base64,{b64}" download="output_video.mp4">Download Output Video</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Output video not found ❌")
        except Exception as e:
            st.error(f"Rendering Failed ❌: {e}")

st.markdown(scroll_script, unsafe_allow_html=True)