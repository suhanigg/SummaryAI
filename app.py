import streamlit as st
import tempfile
import os
import ffmpeg
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Audio → Notes AI",
    page_icon="🎧",
    layout="wide"
)

# Title
st.title("🎧 Audio to Notes AI App")
st.markdown("Upload an MP4 file → Convert speech to text → Generate notes")

# Sidebar
st.sidebar.header("⚙️ Settings")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")

summarize = st.sidebar.checkbox("Enable Notes Summary", value=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload MP4 File", type=["mp4"])

# Main logic
if uploaded_file and hf_token:

    st.success("File uploaded successfully ✅")

    with st.spinner("🔄 Processing... Please wait"):

        # Save temp video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        # Convert to WAV
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True, quiet=True)
        except:
            st.error("❌ FFmpeg not found. Install FFmpeg and add to PATH.")
            st.stop()

        # Load Whisper model
        st.info("🧠 Transcribing audio...")
        speech_to_text = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            use_auth_token=hf_token
        )

        result = speech_to_text(audio_path)
        transcription = result["text"]

        # Display transcription
        st.subheader("📜 Transcription")
        st.text_area("", transcription, height=250)

        # Download transcription
        st.download_button(
            "⬇️ Download Transcription",
            transcription,
            file_name="transcription.txt"
        )

        # Summarization
        if summarize:
            st.info("✍️ Generating notes...")

            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                use_auth_token=hf_token
            )

            summary = summarizer(
                transcription,
                max_length=180,
                min_length=60,
                do_sample=False
            )

            summary_text = summary[0]["summary_text"]

            st.subheader("📝 Notes Summary")
            st.text_area("", summary_text, height=200)

            # Download summary
            st.download_button(
                "⬇️ Download Notes",
                summary_text,
                file_name="notes.txt"
            )

        # Cleanup
        os.remove(video_path)
        os.remove(audio_path)

else:
    st.warning("⚠️ Upload a file and enter Hugging Face token to begin.")
