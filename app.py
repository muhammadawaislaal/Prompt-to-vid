import streamlit as st
import torch
import os
import uuid
import imageio
import tempfile
from PIL import Image
from gtts import gTTS
from tqdm import tqdm
from diffusers import DiffusionPipeline, DDIMScheduler

st.set_page_config(page_title="🎬 Prompt-to-Video Generator", layout="centered")
st.title("🎬 Prompt-to-Video Generator with Voiceover (No Watermark)")

# === Load Stable AnimateDiff Pipeline (public + tested) ===
@st.cache_resource
def load_pipeline():
    model_id = "cerspense/zeroscope_v2_576w"  # Verified public AnimateDiff-compatible model
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_pipeline()

# === UI ===
prompt = st.text_area("🎯 Enter your prompt", height=100, placeholder="A robot dancing under neon lights")
frames = st.slider("🎞️ Number of frames", 16, 64, step=8, value=32)
fps = 8
seed = st.number_input("🔁 Seed (for consistency)", value=42)
submit = st.button("🚀 Generate Video")

if submit and prompt.strip():
    with st.spinner("🧠 Generating frames..."):
        torch.manual_seed(seed)
        output = pipe(prompt=prompt, num_inference_steps=25, num_frames=frames)
        video_tensor = output.frames[0]
        frames_np = (video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        for i, frame in enumerate(tqdm(frames_np)):
            path = os.path.join(temp_dir, f"frame_{i:03}.png")
            Image.fromarray(frame).save(path)
            frame_paths.append(path)

    with st.spinner("🔊 Generating voiceover..."):
        audio_path = os.path.join(temp_dir, "voice.mp3")
        gTTS(prompt).save(audio_path)

    with st.spinner("🎞️ Encoding video..."):
        video_path = os.path.join(temp_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for frame_file in frame_paths:
            writer.append_data(imageio.imread(frame_file))
        writer.close()

    with st.spinner("🔊 Merging audio with video..."):
        final_path = os.path.join(temp_dir, "final_video.mp4")
        ffmpeg_command = (
            f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
        )
        os.system(ffmpeg_command)

    st.success("✅ Video ready!")
    st.video(final_path)
    with open(final_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, file_name="prompt_video.mp4", mime="video/mp4")
