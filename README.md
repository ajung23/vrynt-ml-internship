# Vrynt ML Internship — MLOps, ASR, and Diffusion

This repository documents my internship work, which focused on bridging the gap between AI research and production-ready products.

My main project was to build a full **end-to-end MLOps pipeline** for a Speech-to-Text (ASR) model, deploying it as a scalable cloud service on **AWS SageMaker**.

I also prototyped and built tools for **text-to-image diffusion** and **neural style transfer** to support the company's core product.

---

## Final Products: Streamlit Demo Apps

To prove the success of the pipelines, I built interactive Streamlit apps. These apps allowed the entire team (technical and non-technical) to test the live AI models.

<p align="center">
  <img src="docs/screenshot_streamlit_stt.png" alt="STT Streamlit UI" width="45%"/>
  &nbsp;&nbsp;
  <img src="docs/screenshot_streamlit_diffusion.png" alt="Diffusion Streamlit UI" width="45%"/>
</p>

---

## My Role & Responsibilities

* **MLOps (ASR):** Deployed a `Wav2Vec2` model to a real-time **AWS SageMaker** endpoint, wrote the deployment runbook, and built the Streamlit client to invoke the live API.
* **Prototyping (Diffusion):** Built a parameterized Streamlit app and CLI for reproducible text-to-image generation with CLIP-guided diffusion.
* **Implementation (Style Transfer):** Implemented a VGG-based neural style transfer pipeline for fast style-preset application.
* **Cloud Architecture:** Designed and documented the data flow for an OCI-to-AWS migration, enabling runtime operations.

---

## 1. MLOps: ASR Quickstart (Deploy & Run)

This guide shows how to deploy the ASR model to SageMaker and run the Streamlit app.

### Step 1) Clone & Environment
```bash
git clone [https://github.com/ajung23/vrynt-ml-internship.git](https://github.com/ajung23/vrynt-ml-internship.git)
cd vrynt-ml-internship
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2) Configure AWS Credentials
Use an IAM role with permission for SageMaker, S3, and CloudWatch Logs.
```bash
aws configure
# or set env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
```

### Step 3) Deploy the SageMaker Endpoint
This script deploys the Hugging Face `Wav2Vec2` model.
```bash
python aws/deploy_sagemaker_endpoint.py \
  --model-id facebook/wav2vec2-base-960h \
  --endpoint-name vrynt-stt-demo \
  --instance-type ml.m5.xlarge
```

### Step 4) Run the Streamlit App
This app connects to the endpoint you just deployed.
```bash
streamlit run app/streamlit_app.py -- \
  --endpoint-name vrynt-stt-demo \
  --region us-east-1
```
*(Note: You can also use `--model-id` in the `diffusion_app.py` script to run the text-to-image demo.)*

---

## 2. Research Notebooks (Launch in Colab)

This repo also collects the compact, reproducible notebooks used for the initial research and feasibility testing. Each is self-contained and ready to run on Google Colab.

| Notebook | What it shows | Launch |
|---|---|---|
| **00_wav2vec2_feasibility.ipynb** | Speech-to-text feasibility using HuggingFace `wav2vec2`; quick latency/WER checks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/vrynt-ml-internship/blob/main/notebooks/00_wav2vec2_feasibility.ipynb) |
| **10_clip_diffusion_expts.ipynb** | Text-guided image generation with CLIP guidance; simple ablations/seeds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/vrynt-ml-internship/blob/main/notebooks/10_clip_diffusion_expts.ipynb) |
| **20_style_transfer_vgg.ipynb** | Classic neural style transfer (VGG19 content/style losses) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/vrynt-ml-internship/blob/main/notebooks/20_style_transfer_vgg.ipynb) |

> *Tip: In Colab, set **Runtime → Change runtime type → GPU (T4)** for best performance.*

---

## Technical Details

### ASR Demo: How it Works
1.  User uploads a `.wav` / `.mp3` file in the Streamlit UI.
2.  Client serializes the audio bytes with `application/x-audio` content type.
3.  `boto3` calls the SageMaker endpoint.
4.  The endpoint's `inference.py` script deserializes the audio, runs inference, and returns the transcript JSON.
5.  UI prints the transcript and shows the latency.

### Tech Stack
* **Cloud & MLOps**: AWS SageMaker (endpoint), S3 (artifacts), Boto3
* **AI Models**: `Wav2Vec2` (HF), CLIP-Guided Diffusion, VGG (Style Transfer)
* **App**: Streamlit
* **Ops**: Single-file deploy scripts, structured runbooks, CI

### Repo Layout
```
vrynt-ml-internship/
├─ app/         # Streamlit apps (STT and Diffusion)
├─ aws/         # AWS deployment scripts (SageMaker)
├─ docs/        # Screenshots and architecture notes
├─ notebooks/
│ ├─ 00_wav2vec2_feasibility.ipynb
│ ├─ 10_clip_diffusion_expts.ipynb
│ └─ 20_style_transfer_vgg.ipynb
├─ requirements.txt
└─ README.md
```
