import streamlit as st
import torch
import os
import uuid
from diffusers import AnimateDiffPipeline, DDIMScheduler
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip
from PIL import Image
from tqdm import tqdm
from gtts import gTTS

# === Streamlit Setup ===
st.set_page_config(page_title="AI Prompt-to-Video Generator w/ Voice", layout="centered")
st.title("🎬 Prompt-to-Video Generator with Voiceover (AnimateDiff + gTTS)")

# === Load AnimateDiff (cached) ===
@st.cache_resource
def load_model():
    model_id = "animate-diff/animate-diff-sd15"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_model()

# === UI ===
prompt = st.text_area("🎯 Enter your prompt", height=100, placeholder="A spaceship landing in a neon city")
seed = st.number_input("🎲 Seed (for consistent results)", value=42)
frames = st.slider("🎞️ Number of frames", 16, 64, step=8, value=32)
fps = 8
submit = st.button("Generate Video with Voiceover")

# === Generation ===
if submit and prompt:
    with st.spinner("Generating video frames..."):
        generator = torch.manual_seed(seed)
        output = pipe(prompt, num_inference_steps=25, num_frames=frames, generator=generator)
        frames_tensor = output.frames[0]
        frames_np = (frames_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")

        session_id = uuid.uuid4().hex[:8]
        frame_dir = f"frames_{session_id}"
        os.makedirs(frame_dir, exist_ok=True)
        frame_paths = []

        for i, frame in enumerate(tqdm(frames_np)):
            img = Image.fromarray(frame)
            path = os.path.join(frame_dir, f"frame_{i:03}.png")
            img.save(path)
            frame_paths.append(path)

    with st.spinner("Generating voiceover..."):
        tts = gTTS(text=prompt, lang="en")
        audio_path = os.path.join(frame_dir, "voiceover.mp3")
        tts.save(audio_path)

    with st.spinner("Stitching video and syncing audio..."):
        clip = ImageSequenceClip(frame_paths, fps=fps)
        audio_clip = AudioFileClip(audio_path).set_duration(clip.duration)

        final = clip.set_audio(audio_clip)
        output_path = os.path.join(frame_dir, "final_video.mp4")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    st.success("✅ Video with voiceover generated!")
    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, "prompt_video.mp4", mime="video/mp4")
