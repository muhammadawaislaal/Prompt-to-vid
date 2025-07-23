import streamlit as st
import requests
import tempfile
import os
from gtts import gTTS

st.set_page_config(page_title="🎬 Prompt-to-Video Generator", layout="centered")
st.title("🎬 Prompt-to-Video with Voiceover (Streamlit Secure Token)")

# 🔐 Get token from Streamlit secrets (cloud only)
try:
    HF_TOKEN = st.secrets["huggingface"]["token"]
except KeyError:
    st.error("❌ Hugging Face token not found in secrets. Please set it in Streamlit Cloud.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

prompt = st.text_area("📝 Enter your prompt", "A cat surfing on waves")
submit = st.button("🚀 Generate Video")

if submit and prompt.strip():
    with st.spinner("🎥 Generating video..."):
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

        if response.status_code == 200:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(response.content)

            audio_path = os.path.join(temp_dir, "voice.mp3")
            gTTS(prompt).save(audio_path)

            final_path = os.path.join(temp_dir, "final_video.mp4")
            os.system(
                f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
                f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
            )

            st.success("✅ Video created!")
            st.video(final_path)
            with open(final_path, "rb") as f:
                st.download_button("⬇️ Download", f, file_name="video.mp4", mime="video/mp4")
        else:
            st.error("Failed to generate video. Check your Hugging Face token or usage limits.")
