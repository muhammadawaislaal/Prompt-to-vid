import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import tempfile
import os
from gtts import gTTS
import imageio
from moviepy.editor import ImageSequenceClip, AudioFileClip

st.set_page_config(page_title="🎬 Prompt-to-Animated Video", layout="centered")
st.title("🎬 Prompt-to-Animated Video Generator (Streamlit Cloud Ready)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"💻 Running on: {device.upper()}")

@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipe

pipe = load_pipeline()

prompt = st.text_area("📝 Describe your animation", "A robot dancing in Times Square", height=100)
num_frames = st.slider("🖼️ Number of frames", 8, 32, 16)
fps = st.slider("🎥 FPS", 4, 12, 6)
generate = st.button("🚀 Generate Video")

if generate and prompt:
    with st.spinner("🎨 Generating frames..."):
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        for i in range(num_frames):
            img = pipe(prompt).images[0]
            path = os.path.join(temp_dir, f"frame_{i:03}.png")
            img.save(path)
            frame_paths.append(path)

    with st.spinner("🗣 Generating voiceover..."):
        audio_path = os.path.join(temp_dir, "voice.mp3")
        gTTS(prompt).save(audio_path)

    with st.spinner("🎞 Creating video..."):
        video_path = os.path.join(temp_dir, "video.mp4")
        clip = ImageSequenceClip(frame_paths, fps=fps)
        clip.write_videofile(video_path, audio=False, logger=None)

    with st.spinner("🎙 Merging video and audio..."):
        final_path = os.path.join(temp_dir, "final_video.mp4")
        cmd = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
        os.system(cmd)

    st.success("✅ Video Generated!")
    st.video(final_path)

    with open(final_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, file_name="animated_video.mp4", mime="video/mp4")
