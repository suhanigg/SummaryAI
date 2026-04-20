import streamlit as st
import tempfile
import os
import ffmpeg
from transformers import pipeline

# =========================
# 🔐 HARDCODE TOKEN
# =========================
hf_token = "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"


# =========================
# 🎨 UI CONFIG
# =========================
st.set_page_config(
    page_title="Audio → Notes AI",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 Audio to Notes AI")
st.markdown("Upload **MP4 or WAV** → Convert to text → Generate notes")

# Sidebar
st.sidebar.header("⚙️ Options")
summarize = st.sidebar.checkbox("Generate Notes Summary", value=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload File", type=["mp4", "wav"])


# =========================
# 🚀 MAIN LOGIC
# =========================
if uploaded_file:

    st.success("File uploaded successfully ✅")

    with st.spinner("🔄 Processing... Please wait"):

        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(uploaded_file.read())
            input_path = temp_file.name

        # Convert MP4 → WAV
        if file_ext == "mp4":
            audio_path = input_path.replace(".mp4", ".wav")
            try:
                ffmpeg.input(input_path).output(audio_path).run(overwrite_output=True, quiet=True)
            except Exception as e:
                st.error("❌ FFmpeg error. Make sure packages.txt has ffmpeg.")
                st.stop()
        else:
            audio_path = input_path

        # =========================
        # 🎙️ SPEECH TO TEXT
        # =========================
        st.info("🧠 Transcribing audio...")

        speech_to_text = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            token=hf_token,
            device=-1   # force CPU for Streamlit Cloud
        )

        try:
            result = speech_to_text(audio_path)
            transcription = result["text"]
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            st.stop()

        # Show transcription
        st.subheader("📜 Transcription")
        st.text_area("", transcription, height=250)

        st.download_button(
            "⬇️ Download Transcription",
            transcription,
            file_name="transcription.txt"
        )

        # =========================
        # 📝 SUMMARIZATION
        # =========================
        if summarize:
            st.info("✍️ Generating notes...")

            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                token=hf_token,
                device=-1
            )

            try:
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

            except Exception as e:
                st.error(f"❌ Summarization failed: {e}")

        # =========================
        # 🧹 CLEANUP
        # =========================
        os.remove(input_path)
        if file_ext == "mp4":
            os.remove(audio_path)

else:
    st.warning("⚠️ Please upload an MP4 or WAV file to begin.")
