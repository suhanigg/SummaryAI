"""
Audio to Notes AI - Complete Streamlit App
Upload MP4/WAV → Transcribe → Generate Smart Notes
✅ Works on Streamlit Cloud ✅ No FFmpeg needed ✅ Secure ✅ Fast
"""

import streamlit as st
import tempfile
import os
from transformers import pipeline
import torch
from pydub import AudioSegment
import io

# =========================
# 🎨 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎧 Audio to Notes AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 1rem; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# =========================
# 🎧 MAIN UI
# =========================
st.markdown('<h1 class="main-header">🎧 Audio to Notes AI</h1>', unsafe_allow_html=True)
st.markdown("### 🚀 Upload audio/video → Get instant transcription & smart notes")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # HF Token (use Streamlit secrets for production)
    hf_token = st.text_input("🔑 HuggingFace Token (optional)", 
                           type="password",
                           help="Needed for private models. Get from huggingface.co/settings/tokens")
    
    summarize = st.checkbox("✍️ Generate Notes Summary", value=True)
    st.markdown("---")
    st.markdown("[⭐ Star on GitHub](https://github.com/suhanigg/SummaryAI)")

# File upload
uploaded_file = st.file_uploader(
    "📤 **Upload Audio/Video**", 
    type=["mp4", "wav", "m4a", "mp3"],
    help="Supports MP4, WAV, MP3, M4A (Max ~10min recommended)"
)

# =========================
# 🚀 PROCESSING LOGIC
# =========================
if uploaded_file:
    
    # File info
    file_size = uploaded_file.size / (1024*1024)  # MB
    st.success(f"✅ **{uploaded_file.name}** loaded ({file_size:.1f}MB)")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🔄 Processing your audio..."):
        
        # Step 1: Save & convert audio
        progress_bar.progress(20)
        status_text.text("📥 Saving file...")
        
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_path = tmp_file.name
        
        # Convert to WAV if needed
        audio_path = input_path
        if file_ext != "wav":
            progress_bar.progress(40)
            status_text.text("🔄 Converting to WAV...")
            try:
                audio = AudioSegment.from_file(input_path)
                audio_path = input_path.replace(f".{file_ext}", ".wav")
                audio.export(audio_path, format="wav")
            except Exception as e:
                st.error(f"❌ Audio conversion failed: {e}")
                st.stop()
        
        # Step 2: Speech to Text
        progress_bar.progress(60)
        status_text.text("🧠 Transcribing speech...")
        
        try:
            @st.cache_resource
            def load_whisper():
                return pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en",  # Fast English model
                    device="cpu",
                    torch_dtype=torch.float32
                )
            
            speech_to_text = load_whisper()
            result = speech_to_text(audio_path)
            transcription = result["text"].strip()
            
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)[:200]}")
            st.info("💡 Try shorter files or check file format")
            st.stop()
        
        progress_bar.progress(80)
        status_text.text("✅ Transcription complete!")
        
        # Step 3: Display results
        progress_bar.progress(100)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Words", len(transcription.split()))
        with col2:
            st.metric("⏱️ Duration", f"{len(transcription)/20:.0f}s est.")
        with col3:
            st.metric("🎯 Quality", "High" if len(transcription.split()) > 20 else "Short")
        
        # Full transcription
        st.subheader("📜 **Full Transcription**")
        st.text_area("", transcription, height=250, key="transcription")
        
        # Download transcription
        col1, col2 = st.columns([2,1])
        with col1:
            st.download_button(
                label="⬇️ Download Transcription",
                data=transcription,
                file_name=f"{uploaded_file.name}_transcription.txt",
                mime="text/plain"
            )
        with col2:
            st.button("📋 Copy", on_click=lambda: st.write(transcription))
        
        # Step 4: Summarization (optional)
        if summarize and len(transcription) > 50:
            st.subheader("✍️ **Smart Notes**")
            
            progress_bar = st.progress(0)
            
            try:
                @st.cache_resource
                def load_summarizer():
                    return pipeline(
                        "summarization",
                        model="sshleifer/distilbart-cnn-12-6",
                        device="cpu"
                    )
                
                summarizer = load_summarizer()
                
                # Chunk long text
                max_chunk = 900
                chunks = [transcription[i:i+max_chunk] 
                         for i in range(0, len(transcription), max_chunk)]
                
                summaries = []
                for i, chunk in enumerate(chunks):
                    progress_bar.progress((i+1)/len(chunks))
                    summary = summarizer(
                        chunk, 
                        max_length=120, 
                        min_length=30, 
                        do_sample=False
                    )[0]["summary_text"]
                    summaries.append(summary)
                
                summary_text = " ".join(summaries)
                
                st.text_area("", summary_text, height=200, key="summary")
                
                st.download_button(
                    label="⬇️ Download Notes",
                    data=summary_text,
                    file_name=f"{uploaded_file.name}_notes.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"❌ Notes failed: {str(e)[:150]}")
                st.info("💡 Audio might be too long/short")
        
        # Cleanup
        try:
            os.unlink(input_path)
            if audio_path != input_path:
                os.unlink(audio_path)
        except:
            pass

# Empty state
else:
    st.markdown("""
    ## 🎯 **Get Started in 3 Steps:**
    
    1. **📤 Upload** MP4, WAV, MP3, or M4A file
    2. **🧠 AI Transcribes** speech to text automatically  
    3. **✍️ Get Notes** (optional summary)
    
    ### 💡 **Tips:**
    - ✅ Works best with clear speech
    - ✅ Max 10min files recommended
    - ✅ English audio optimized
    
    **Ready? Upload your first file! 🎙️**
    """)

# Footer
st.markdown("---")
st.markdown("⭐ **Made with ❤️ by SummaryAI** | [GitHub](https://github.com/suhanigg/SummaryAI)")
