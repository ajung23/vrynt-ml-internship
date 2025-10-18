# diffusion_app.py
import argparse, os, time
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_pipeline(model_id: str):
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device), device

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    args, _ = parser.parse_known_args()

    st.title("Diffusion Demo - Text to Image")
    st.caption("Lean prototype UI for prompts -> preview")

    prompt = st.text_input("Prompt", "product mockup, clean studio lighting, neutral background")
    seed = st.number_input("Seed", value=1234, step=1)
    steps = st.slider("Steps", min_value=10, max_value=50, value=30)
    guidance = st.slider("Guidance", min_value=3.5, max_value=12.0, value=7.5, step=0.5)
    model_id = st.text_input("Model ID", args.model_id)

    if st.button("Generate"):
        pipe, device = load_pipeline(model_id)
        g = torch.Generator(device=device).manual_seed(int(seed))
        t0 = time.time()
        out = pipe(prompt, num_inference_steps=int(steps), guidance_scale=float(guidance), generator=g)
        img = out.images[0]
        latency = time.time() - t0
        st.image(img, caption=f"{model_id} - {latency:.2f}s")
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", f"out_{int(time.time())}.png")
        img.save(path)
        st.success(f"Saved to {path}")

if __name__ == "__main__":
    main()
