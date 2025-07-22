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

st.set_page_config(page_title="Prompt-to-Video w/ Voice", layout="centered")
st.title("🎬 Prompt-to-Video Generator (High Quality + Voice)")

# === Load AnimateDiff ===
@st.cache_resource
def load_model():
    model_id = "animate-diff/animate-diff-sd15"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_model()

prompt = st.text_area("🎯 Prompt", height=100, placeholder="A futuristic city during sunset with flying cars")
frames = st.slider("🎞️ Frames", 16, 64, 8, 32)
fps = 8
seed = st.number_input("🔁 Seed", value=42)
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

    with st.spinner("🎞️ Creating video from frames..."):
        video_path = os.path.join(temp_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for f in frame_paths:
            writer.append_data(imageio.imread(f))
        writer.close()

    with st.spinner("🔊 Merging video with audio (ffmpeg)..."):
        final_output_path = os.path.join(temp_dir, "final_output.mp4")
        os.system(f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v libx264 -c:a aac -shortest "{final_output_path}"')

    st.success("✅ Done!")
    st.video(final_output_path)
    with open(final_output_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, "prompt_video.mp4", mime="video/mp4")
