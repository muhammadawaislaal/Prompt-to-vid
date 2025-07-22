import streamlit as st
import torch
import os
import tempfile
import imageio
from PIL import Image
from gtts import gTTS
from diffusers import DiffusionPipeline

st.set_page_config(page_title="📝 Prompt-to-Video Generator", layout="centered")
st.title("🎬 Prompt-to-Video with Voiceover (Streamlit Cloud Safe)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"💻 Running on: {device.upper()}")

@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float32
    ).to(device)
    return pipe

pipe = load_pipeline()

prompt = st.text_area("🎯 Enter a prompt to animate", "A cat surfing a wave during sunset")
fps = 6
submit = st.button("🚀 Generate Video")

if submit and prompt.strip():
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info("🧠 Generating video from prompt...")
        result = pipe(prompt)
        frames = result.frames[0]
        
        video_path = os.path.join(temp_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        st.info("🔊 Generating voiceover...")
        audio_path = os.path.join(temp_dir, "voice.mp3")
        gTTS(prompt).save(audio_path)

        st.info("🔄 Merging audio with video...")
        final_path = os.path.join(temp_dir, "final_output.mp4")
        os.system(
            f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
        )

        st.success("✅ All done!")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("⬇️ Download Final Video", f, "prompt_video.mp4", mime="video/mp4")
