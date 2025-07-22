import streamlit as st
import torch
import os
import tempfile
from gtts import gTTS
import imageio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

st.set_page_config(page_title="🎬 Prompt-to-Video Generator", layout="centered")
st.title("🎬 Prompt-to-Video Generator with Voiceover (No Watermark)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"💻 Running on: {device.upper()}")

@st.cache_resource
def load_pipeline():
    return pipeline(
        task=Tasks.text_to_video_synthesis,
        model='damo-vilab/text-to-video-ms-1.7b',
        device=torch.device(device)
    )

pipe = load_pipeline()

prompt = st.text_area("🎯 Enter your prompt", height=100, placeholder="A panda surfing on a futuristic hoverboard in space")
submit = st.button("🚀 Generate Video")

if submit and prompt.strip():
    with st.spinner("🎥 Generating video from prompt..."):
        result = pipe({'text': prompt})
        video_path = result['output_video']

    with st.spinner("🔊 Generating voiceover..."):
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "voice.mp3")
        gTTS(prompt).save(audio_path)

    with st.spinner("🎞️ Merging voiceover with video..."):
        final_path = os.path.join(temp_dir, "final_video.mp4")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
        )
        os.system(ffmpeg_cmd)

    st.success("✅ Done! Here’s your video:")
    st.video(final_path)
    with open(final_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, "prompt_video.mp4", mime="video/mp4")
