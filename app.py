import streamlit as st
import torch
import os
import tempfile
import imageio
from PIL import Image
from gtts import gTTS
from diffusers import StableVideoDiffusionPipeline

st.set_page_config(page_title="🎬 Image-to-Video Generator", layout="centered")
st.title("🎬 Image-to-Video Generator with Voiceover (Stable Video Diffusion)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"💻 Running on: {device.upper()}")

@st.cache_resource
def load_pipeline():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_pipeline()

uploaded_image = st.file_uploader("🖼 Upload an input image", type=["jpg", "jpeg", "png"])
caption = st.text_input("🎤 Enter a description (used for voiceover)", value="A dreamy cinematic moment.")

submit = st.button("🚀 Generate Video")

if submit and uploaded_image and caption.strip():
    with tempfile.TemporaryDirectory() as temp_dir:
        image = Image.open(uploaded_image).convert("RGB").resize((512, 512))
        image_path = os.path.join(temp_dir, "input.png")
        image.save(image_path)

        with st.spinner("🎥 Generating video from image..."):
            video_frames = pipe(image, decode_chunk_size=8, generator=torch.manual_seed(42)).frames[0]
            frame_paths = []
            for i, frame in enumerate(video_frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:03}.png")
                frame.save(frame_path)
                frame_paths.append(frame_path)

            video_path = os.path.join(temp_dir, "video.mp4")
            writer = imageio.get_writer(video_path, fps=7)
            for f in frame_paths:
                writer.append_data(imageio.imread(f))
            writer.close()

        with st.spinner("🔊 Generating voiceover..."):
            audio_path = os.path.join(temp_dir, "voice.mp3")
            gTTS(caption).save(audio_path)

        with st.spinner("🔄 Merging voice with video..."):
            final_path = os.path.join(temp_dir, "final_video.mp4")
            os.system(
                f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
                f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
            )

        st.success("✅ Done! Preview your video below:")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("⬇️ Download Video", f, "dream_video.mp4", mime="video/mp4")
