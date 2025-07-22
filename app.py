import streamlit as st
import torch
import os
import tempfile
import imageio
from PIL import Image
from gtts import gTTS
from diffusers import DiffusionPipeline, DDIMScheduler

st.set_page_config(page_title="🎬 Prompt-to-Video Generator", layout="centered")
st.title("🎬 Prompt-to-Video with Voiceover (Streamlit Cloud-Compatible)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"💻 Running on: {device.upper()}")

@st.cache_resource
def load_pipeline():
    dtype = torch.float32 if device == "cpu" else torch.float16
    model_id = "cerspense/zeroscope_v2_576w"
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        revision="fp16" if dtype == torch.float16 else "main"
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

pipe = load_pipeline()

prompt = st.text_area("📝 Enter your video prompt", "A robot dancing in the rain under neon lights")
frames = st.slider("🎞️ Number of frames", 16, 64, 32, step=8)
fps = 8
submit = st.button("🚀 Generate Video")

if submit and prompt.strip():
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info("⏳ Generating frames from prompt...")
        torch.manual_seed(42)
        output = pipe(prompt=prompt, num_inference_steps=25, num_frames=frames)
        frames_tensor = output.frames[0].permute(0, 2, 3, 1).cpu().numpy()
        frames_uint8 = (frames_tensor * 255).astype("uint8")

        frame_paths = []
        for i, frame in enumerate(frames_uint8):
            frame_path = os.path.join(temp_dir, f"frame_{i:03}.png")
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)

        video_path = os.path.join(temp_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=fps)
        for f in frame_paths:
            writer.append_data(imageio.imread(f))
        writer.close()

        st.info("🎤 Generating voiceover...")
        audio_path = os.path.join(temp_dir, "voice.mp3")
        gTTS(prompt).save(audio_path)

        st.info("🎬 Merging audio and video...")
        final_path = os.path.join(temp_dir, "final_output.mp4")
        os.system(
            f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
            f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
        )

        st.success("✅ Done! Preview your result:")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("⬇️ Download Video", f, "generated_video.mp4", mime="video/mp4")
