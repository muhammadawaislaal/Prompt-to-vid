import streamlit as st
import torch
import os
import uuid
from diffusers import AnimateDiffPipeline, DDIMScheduler
from moviepy.editor import ImageSequenceClip
from PIL import Image
from tqdm import tqdm

# === Setup Streamlit ===
st.set_page_config(page_title="Prompt-to-Video Generator", layout="centered")
st.title("🎬 Custom Prompt-to-Video Generator (High-Quality, No Watermark)")

# === Load AnimateDiff Pipeline (cached) ===
@st.cache_resource
def load_model():
    model_id = "animate-diff/animate-diff-sd15"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_model()

# === UI ===
prompt = st.text_area("Enter your prompt", height=100, placeholder="A tiger walking through neon jungle at night")
seed = st.number_input("Random seed (set for consistent results)", value=42)
frames = st.slider("Number of frames", min_value=16, max_value=64, step=8, value=32)
submit = st.button("Generate Video")

# === Generation Logic ===
if submit and prompt:
    with st.spinner("Generating frames..."):
        generator = torch.manual_seed(seed)
        output = pipe(prompt, num_inference_steps=25, num_frames=frames, generator=generator)
        frames_tensor = output.frames[0]  # (frames, C, H, W)
        frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
        
        img_dir = f"frames_{uuid.uuid4().hex[:6]}"
        os.makedirs(img_dir, exist_ok=True)
        img_paths = []
        
        for i, frame in enumerate(tqdm(frames_np)):
            img = Image.fromarray(frame)
            path = os.path.join(img_dir, f"frame_{i:03}.png")
            img.save(path)
            img_paths.append(path)
        
        # Create video
        st.spinner("Stitching into video...")
        clip = ImageSequenceClip(img_paths, fps=8)
        video_path = os.path.join(img_dir, "generated_video.mp4")
        clip.write_videofile(video_path, codec="libx264", audio=False)

        st.success("✅ Video created!")
        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button("Download Video", f, "prompt_video.mp4", mime="video/mp4")
