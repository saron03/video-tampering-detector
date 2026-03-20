import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.detect_video import detect_video


st.set_page_config(page_title="Video Tampering Detector", layout="wide")


def save_uploaded_file(uploaded_file):
    temp_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "temp"
    )
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def render_traditional_details(result):
    st.subheader("Traditional Method Details")

    fc = result.get("frame_correlation", {})
    of = result.get("optical_flow", {})
    cm = result.get("copy_move", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Frame Correlation", fc.get("prediction", "N/A"))
    c2.metric("Optical Flow", of.get("prediction", "N/A"))
    c3.metric("Copy-Move", cm.get("prediction", "N/A"))

    st.write("Scores")
    st.progress(min(max(float(fc.get("score", 0.0)), 0.0), 1.0), text=f"Frame Correlation: {fc.get('score', 0.0):.4f}")
    st.progress(min(max(float(of.get("score", 0.0)), 0.0), 1.0), text=f"Optical Flow: {of.get('score', 0.0):.4f}")
    st.progress(min(max(float(cm.get("score", 0.0)), 0.0), 1.0), text=f"Copy-Move: {cm.get('score', 0.0):.4f}")

    if result.get("reasoning"):
        st.info(result.get("reasoning"))


def render_modern_details(result):
    st.subheader("CNN Details")
    st.metric("Fake Probability", f"{result.get('score', 0.0) * 100:.2f}%")

    frame_probs = result.get("frame_probs", [])
    if frame_probs:
        st.line_chart(frame_probs)


def main():
    st.title("Video Tampering Detection")
    st.caption("Upload a video, choose a method, and run analysis.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"]
        )

        method = st.selectbox(
            "Detection method",
            options=["traditional", "modern"],
            format_func=lambda x: "Traditional" if x == "traditional" else "Modern (CNN)"
        )

        max_frames = st.slider("Max frames", min_value=10, max_value=200, value=50, step=10)
        run_analysis = st.button("Run Analysis", use_container_width=True)

        st.markdown("---")
        st.write("Method summary")
        if method == "traditional":
            st.write("Uses frame correlation, optical flow, and copy-move analysis.")
        else:
            st.write("Uses the trained EfficientNet CNN for deepfake detection.")

    with right:
        if not uploaded_file:
            st.info("Upload a video file to start.")
            return

        st.write(f"Selected file: {uploaded_file.name}")
        st.write(f"Size: {uploaded_file.size / 1024 / 1024:.2f} MB")

        if not run_analysis:
            st.warning("Click 'Run Analysis' to process this video.")
            return

        temp_path = save_uploaded_file(uploaded_file)

        try:
            with st.spinner("Running detection..."):
                result = detect_video(
                    video_path=temp_path,
                    method=method,
                    max_frames=max_frames,
                    cleanup=True
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if "error" in result:
            st.error(result["error"])
            return

        st.success("Analysis complete")

        prediction = result.get("prediction", "Unknown")
        if prediction == "Tampered":
            st.error("Verdict: TAMPERED")
        elif prediction == "Authentic":
            st.success("Verdict: AUTHENTIC")
        else:
            st.info(f"Verdict: {prediction}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", prediction)
        c2.metric("Confidence", f"{result.get('confidence', 0):.2f}%")
        c3.metric("Method", result.get("method", "").capitalize())

        if method == "traditional":
            render_traditional_details(result)
        else:
            render_modern_details(result)

        with st.expander("Raw result"):
            st.json(result)


if __name__ == "__main__":
    main()