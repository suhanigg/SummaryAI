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
st.markdown("Upload MP4 or WAV → Convert speech to text → Generate notes")

# Sidebar
st.sidebar.header("⚙️ Settings")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")
summarize = st.sidebar.checkbox("Enable Notes Summary", value=True)

# File upload (UPDATED)
uploaded_file = st.file_uploader("📤 Upload Audio/Video File", type=["mp4", "wav"])

if uploaded_file and hf_token:

    st.success("File uploaded successfully ✅")

    with st.spinner("🔄 Processing... Please wait"):

        file_ext = uploaded_file.name.split(".")[-1]

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(uploaded_file.read())
            input_path = temp_file.name

        # If MP4 → convert to WAV
        if file_ext == "mp4":
            audio_path = input_path.replace(".mp4", ".wav")
            try:
                ffmpeg.input(input_path).output(audio_path).run(overwrite_output=True, quiet=True)
            except:
                st.error("❌ FFmpeg error. Make sure it's installed.")
                st.stop()
        else:
            # If already WAV → no conversion
            audio_path = input_path

        # Transcription
        st.info("🧠 Transcribing audio...")
        speech_to_text = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            use_auth_token=hf_token
        )

        result = speech_to_text(audio_path)
        transcription = result["text"]

        # Show transcription
        st.subheader("📜 Transcription")
        st.text_area("", transcription, height=250)

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

            st.download_button(
                "⬇️ Download Notes",
                summary_text,
                file_name="notes.txt"
            )

        # Cleanup
        os.remove(input_path)
        if file_ext == "mp4":
            os.remove(audio_path)

else:
    st.warning("⚠️ Upload a file and enter Hugging Face token to begin.")
