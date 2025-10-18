# streamlit_app.py
import argparse, io, json, time
import streamlit as st
import boto3
from botocore.config import Config
from pydub import AudioSegment

def transcribe(endpoint_name, region, audio_bytes):
    smr = boto3.client("sagemaker-runtime", region_name=region, config=Config(retries={'max_attempts': 3}))
    t0 = time.time()
    resp = smr.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-audio",
        Body=audio_bytes,
    )
    latency = time.time() - t0
    body = resp["Body"].read()
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        data = {"raw": body.decode("utf-8", errors="ignore")}
    return data, latency

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    args, _ = parser.parse_known_args()
    st.title("Vrynt — STT Demo (Wav2Vec2 on SageMaker)")
    st.caption("Upload an audio file and get a transcript from a SageMaker endpoint.")
    upl = st.file_uploader("Audio (.wav/.mp3)", type=["wav", "mp3"])
    if upl and st.button("Transcribe"):
        # normalize to wav bytes for safer server handling
        audio = AudioSegment.from_file(upl, format=upl.name.split(".")[-1])
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        data, latency = transcribe(args.endpoint_name, args.region, buf.getvalue())
        st.write("**Transcript:**")
        st.json(data)
        st.write(f"Latency: {latency:.2f}s")

if __name__ == "__main__":
    main()
