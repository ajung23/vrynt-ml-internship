# Vrynt ML Internship: MLOps & OCI-to-AWS Migration

This repository documents my internship, which was focused on a critical business objective: **evaluating and migrating the company's AI infrastructure from Oracle Cloud (OCI) to Amazon Web Services (AWS).**

My primary role was to act as an MLOps engineer, proving the viability of AWS for the company's AI workloads. I did this by building a complete, end-to-end pipeline for a real-time Speech-to-Text (ASR) model, deploying it as a scalable cloud service on **AWS SageMaker**.

I was responsible for the entire project lifecycle: from initial evaluation and cloud architecture design to deploying the final application and writing the comprehensive migration documentation for the rest of the engineering team.

---

## Final Products: Deployed Endpoint & Streamlit Demo

To prove the success of the migration, I built a production-grade ASR service on SageMaker and a front-end Streamlit app to interact with it. This allowed the entire team to test the performance, cost, and stability of the new AWS-based infrastructure.

<p align="center">
  <img src="docs/screenshot_streamlit_stt.png" alt="STT Streamlit UI" width="45%"/>
  &nbsp;&nbsp;
  <img src="docs/screenshot_streamlit_diffusion.png" alt="Diffusion Streamlit UI" width="45%"/>
</p>

---

## My Role & Responsibilities

* **Cloud Migration (OCI → AWS):** Evaluated OCI services and designed a migration path to AWS, focusing on building a scalable, robust, and cost-effective solution.
* **MLOps (ASR):** Deployed a `Wav2Vec2` model to a real-time **AWS SageMaker** endpoint and wrote the deployment runbook.
* **Application:** Built the Streamlit client to invoke the live SageMaker API, handling real-time data streams.
* **Documentation:** Created comprehensive documentation detailing the entire migration process, from evaluation to deployment, to ensure consistency and quality for the team.
* **Prototyping:** I also prototyped text-to-image (Diffusion) and style-transfer models to support other product goals.

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

---

## 2. Research Notebooks (Launch in Colab)

These are the reproducible notebooks I used for initial research and feasibility testing for the company's various AI initiatives.

| Notebook | What it shows | Launch |
|---|---|---|
| **00_wav2vec2_feasibility.ipynb** | Speech-to-text feasibility (latency/WER checks) before SageMaker deployment. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/vrynt-ml-internship/blob/main/notebooks/00_wav2vec2_feasibility.ipynb) |
| **10_clip_diffusion_expts.ipynb** | Text-guided image generation (CLIP) research for product prototyping. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/vrynt-ml-internship/blob/main/notebooks/10_clip_diffusion_expts_fixed.ipynb) |
| **20_style_transfer_vgg.ipynb** | Classic neural style transfer (VGG) for fast style-preset application. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajung23/vrynt-ml-internship/blob/main/notebooks/20_style_transfer_vgg.ipynb) |

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
* **Cloud & MLOps**: AWS SageMaker, S3 (artifacts), CloudWatch Logs, Boto3
* **AI Models**: `Wav2Vec2` (HF), CLIP-Guided Diffusion, VGG (Style Transfer)
* **App & Data**: Streamlit, Pandas, Librosa, Jiwer (for WER/CER)
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
├─ outputs/     # Generated images from notebooks
├─ requirements.txt
└─ README.md
```
