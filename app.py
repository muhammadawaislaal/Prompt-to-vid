import streamlit as st
import torch
import os
import uuid
from diffusers import AnimateDiffPipeline, DDIMScheduler
from moviepy.editor import ImageSequenceClip, AudioFileClip
from PIL import Image
from tqdm import tqdm
from gtts import gTTS
import tempfile

# === Streamlit Setup ===
st.set_page_config(page_title="Prompt-to-Video with Voice", layout="centered")
st.title("🎬 AI Prompt-to-Video Generator with Voiceover")

# === Load AnimateDiff Model (cached) ===
@st.cache_resource
def load_model():
    model_id = "animate-diff/animate-diff-sd15"
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_model()

# === UI Inputs ===
prompt = st.text_area("Enter your prompt", height=100, placeholder="A robot dancing in Times Square at night")
seed = st.number_input("Seed (optional)", value=42)
frames = st.slider("Frames", 16, 64, step=8, value=32)
fps = 8

if st.button("Generate Video"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating video frames..."):
            generator = torch.manual_seed(seed)
            output = pipe(prompt, num_inference_steps=25, num_frames=frames, generator=generator)
            frames_tensor = output.frames[0]
            frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

            temp_dir = tempfile.mkdtemp()
            frame_paths = []
            for i, frame in enumerate(tqdm(frames_np)):
                img = Image.fromarray(frame)
                path = os.path.join(temp_dir, f"frame_{i:03}.png")
                img.save(path)
                frame_paths.append(path)

        with st.spinner("Generating voiceover..."):
            audio_path = os.path.join(temp_dir, "voiceover.mp3")
            tts = gTTS(text=prompt, lang="en")
            tts.save(audio_path)

        with st.spinner("Rendering final video..."):
            clip = ImageSequenceClip(frame_paths, fps=fps)
            audio = AudioFileClip(audio_path).set_duration(clip.duration)
            final = clip.set_audio(audio)
            output_path = os.path.join(temp_dir, "final_video.mp4")
            final.write_videofile(output_path, codec="libx264", audio_codec="aac")

        st.success("✅ Video ready!")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Video", f, "prompt_video.mp4", mime="video/mp4")
