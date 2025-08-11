import streamlit as st
import tempfile
import cv2
import os
import numpy as np
from typing import List
import time

from .pose_estimation import process_video, PoseEstimator
from .models import LiftData, FrameData, JointAngles
from .rule_checker import RuleChecker, extract_joint_angles
from .utils import overlay_skeleton_and_annotations, get_key_frames, create_report

def main():
    st.set_page_config(page_title="AI Powerlifting Judge", layout="wide")

    st.title("AI-Powered Powerlifting Lift Assessor")
    st.markdown("""
    This app analyzes powerlifting attempts using computer vision to check compliance with IPF rules.
    Upload a video of your lift, and the AI will assess if it meets competition standards.
    """)

    # Sidebar
    st.sidebar.title("Settings")
    lift_type = st.sidebar.selectbox(
        "Select Lift Type",
        ["Squat", "Bench Press", "Deadlift"],
        help="Choose the type of lift you want to analyze"
    ).lower().replace(" ", "")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your lift video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        help="Upload a video showing your complete lift from start to finish"
    )

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        with st.spinner("Processing video... This may take a minute."):
            try:
                # Process video
                progress_bar = st.progress(0)
                landmarks_sequence = process_video(video_path, desired_fps=15)
                progress_bar.progress(33)

                # Build LiftData
                frames = []
                for lm in landmarks_sequence:
                    angles_dict = extract_joint_angles(lm)
                    frame_data = FrameData(
                        landmarks=lm,
                        angles=JointAngles(**angles_dict)
                    )
                    frames.append(frame_data)
                lift_data = LiftData(frames=frames)
                progress_bar.progress(66)

                # Assess lift
                result = RuleChecker.assess_lift(lift_type, lift_data)
                progress_bar.progress(100)
                time.sleep(0.5)  # Let user see completion
                progress_bar.empty()

                # Display results in columns
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Display key frames
                    st.subheader("Key Frames Analysis")
                    key_frames = get_key_frames(video_path, result.frame_indices)
                    
                    for name, frame in key_frames.items():
                        frame_idx = result.frame_indices[name]
                        landmarks = landmarks_sequence[frame_idx] if frame_idx < len(landmarks_sequence) else None
                        annotated = overlay_skeleton_and_annotations(
                            frame,
                            landmarks,
                            name.replace('_', ' ').title()
                        )
                        st.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=name.replace('_', ' ').title(),
                            use_column_width=True
                        )

                with col2:
                    # Display verdict and infractions
                    st.subheader("Lift Assessment")
                    if result.is_good_lift:
                        st.success("🎉 GOOD LIFT!")
                    else:
                        st.error("❌ NO LIFT")
                        st.subheader("Infractions:")
                        for inf in result.infractions:
                            st.warning(f"- {inf}")

                    # Display joint angles
                    st.subheader("Joint Angles at Key Points")
                    for name, idx in result.frame_indices.items():
                        if idx < len(frames):
                            frame = frames[idx]
                            st.write(f"**{name.replace('_', ' ').title()}:**")
                            angles = {
                                "Hip": f"{frame.angles.hip_angle:.1f}°" if frame.angles.hip_angle else "N/A",
                                "Knee": f"{frame.angles.knee_angle:.1f}°" if frame.angles.knee_angle else "N/A",
                                "Shoulder": f"{frame.angles.shoulder_angle:.1f}°" if frame.angles.shoulder_angle else "N/A",
                                "Elbow": f"{frame.angles.elbow_angle:.1f}°" if frame.angles.elbow_angle else "N/A",
                            }
                            for joint, angle in angles.items():
                                st.text(f"{joint}: {angle}")
                            st.write("---")

                    # Allow downloading a summary report
                    report_path = os.path.join(tempfile.gettempdir(), "lift_report.png")
                    create_report(result, key_frames, report_path)
                    with open(report_path, "rb") as report_file:
                        st.download_button(
                            label="Download Analysis Report",
                            data=report_file,
                            file_name="lift_report.png",
                            mime="image/png",
                        )
                    os.remove(report_path)

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.write("Please ensure the video shows the complete lift clearly.")
            finally:
                # Cleanup
                os.unlink(video_path)

    # Instructions
    with st.expander("How to Use"):
        st.markdown("""
        1. Select your lift type from the sidebar
        2. Upload a video of your lift
        3. Wait for the AI to process and analyze the lift
        4. Review the results, including:
           - Key frame analysis with pose detection
           - Joint angles at critical points
           - Any rule infractions detected
        
        **Tips for Best Results:**
        - Record from a side angle
        - Ensure good lighting
        - Wear fitted clothing
        - Keep the camera stable
        - Include the complete lift from setup to lockout
        """)

    # About
    with st.expander("About"):
        st.markdown("""
        This app uses computer vision and pose estimation to analyze powerlifting attempts
        according to IPF (International Powerlifting Federation) technical rules.
        
        **Technologies Used:**
        - MediaPipe for pose estimation
        - OpenCV for video processing
        - Streamlit for the web interface
        
        **Limitations:**
        - The analysis is approximate and should not replace human judges
        - Accuracy depends on video quality and camera angle
        - Some rules (like foot movement) may be hard to detect
        
        For competition purposes, always refer to official IPF judges and rulebook.
        """)

if __name__ == "__main__":
    main() 