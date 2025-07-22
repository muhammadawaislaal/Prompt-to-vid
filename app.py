import streamlit as st
import torch
import os
import uuid
import imageio
import tempfile
from PIL import Image
from gtts import gTTS
from diffusers import DiffusionPipeline, DDIMScheduler
from tqdm import tqdm
from huggingface_hub import login

# Optional: Hugging Face token (only if private/protected model access is needed)
if "HUGGINGFACE_TOKEN" in st.secrets:
    login(token=st.secrets["HUGGINGFACE_TOKEN"])

st.set_page_config(page_title="Prompt-to-Video Generator", layout="centered")
st.title("🎬 Prompt-to-Video Generator with Voice (No Watermark)")

# === Load Model (zeroscope) ===
@st.cache_resource
def load_pipeline():
    model_id = "cerspense/zeroscope_v2_576w"
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_pipeline()

# === UI ===
prompt = st.text_area("Enter your prompt", height=100, placeholder="A dragon flying over a neon city at night")
frames = st.slider("🎞️ Number of frames", 16, 64, step=8, value=32)
fps = 8
seed = st.number_input("Random seed", value=42)
submit = st.button("Generate Video")

# === Generation Logic ===
if submit and prompt:
    with st.spinner("🎥 Generating video frames..."):
        torch.manual_seed(seed)
        video_output = pipe(prompt=prompt, num_inference_steps=25, num_frames=frames)
        frames_tensor = video_output.frames[0]
        frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        for i, frame in enumerate(tqdm(frames_np)):
            img = Image.fromarray(frame)
            path = os.path.join(temp_dir, f"frame_{i:03}.png")
            img.save(path)
            frame_paths.append(path)

    with st.spinner("🎙️ Generating voiceover..."):
        audio_path = os.path.join(temp_dir, "voice.mp3")
        tts = gTTS(prompt)
        tts.save(audio_path)

    with st.spinner("🧵 Stitching video..."):
        video_path = os.path.join(temp_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for f in frame_paths:
            writer.append_data(imageio.imread(f))
        writer.close()

    with st.spinner("🔊 Merging audio with video..."):
        final_output_path = os.path.join(temp_dir, "final_output.mp4")
        os.system(f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v libx264 -c:a aac -shortest "{final_output_path}"')

    st.success("✅ Done!")
    st.video(final_output_path)
    with open(final_output_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, "prompt_video.mp4", mime="video/mp4")
