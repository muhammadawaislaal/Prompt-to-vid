import streamlit as st
import torch
import os
import uuid
import imageio
import tempfile
from PIL import Image
from gtts import gTTS
from diffusers import AnimateDiffPipeline, DDIMScheduler
from tqdm import tqdm
from pydub import AudioSegment
from pydub.generators import Silent

st.set_page_config(page_title="Prompt-to-Video w/ Voice", layout="centered")
st.title("🎬 Prompt-to-Video Generator (Voiceover, No Watermark)")

# === Load AnimateDiff (cached) ===
@st.cache_resource
def load_model():
    model_id = "animate-diff/animate-diff-sd15"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_model()

prompt = st.text_area("🎯 Enter your prompt", height=100, placeholder="A robot dancing on Mars at night")
frames = st.slider("🎞️ Number of frames", 16, 64, step=8, value=32)
fps = 8
seed = st.number_input("🔁 Seed (optional)", value=42)
submit = st.button("Generate Video")

if submit and prompt:
    with st.spinner("🧠 Generating video frames..."):
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

    with st.spinner("🎙️ Generating voiceover..."):
        audio_path = os.path.join(temp_dir, "voice.mp3")
        tts = gTTS(prompt)
        tts.save(audio_path)

    with st.spinner("🎞️ Stitching video..."):
        video_path = os.path.join(temp_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=fps)

        for f in frame_paths:
            writer.append_data(imageio.imread(f))
        writer.close()

    with st.spinner("🔊 Merging audio and video..."):
        video = AudioSegment.silent(duration=len(frame_paths) * 1000 // fps)
        voice = AudioSegment.from_mp3(audio_path)

        combined_audio = voice + Silent(duration=max(0, len(video) - len(voice)))
        final_audio_path = os.path.join(temp_dir, "combined.mp3")
        combined_audio.export(final_audio_path, format="mp3")

        final_output_path = os.path.join(temp_dir, "final_output.mp4")
        os.system(f'ffmpeg -y -i "{video_path}" -i "{final_audio_path}" -c:v copy -c:a aac -strict experimental "{final_output_path}"')

    st.success("✅ Done! Here's your video with voiceover:")
    st.video(final_output_path)
    with open(final_output_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, "prompt_video.mp4", mime="video/mp4")
