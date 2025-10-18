# scripts/transcribe_local.py
# Utility to hit a deployed SageMaker endpoint from the CLI.
import argparse, json, io
import boto3
from botocore.config import Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--audio", required=True, help="path to wav/mp3")
    args = ap.parse_args()

    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    smr = boto3.client("sagemaker-runtime", region_name=args.region, config=Config(retries={'max_attempts': 3}))
    resp = smr.invoke_endpoint(EndpointName=args.endpoint_name, ContentType="application/x-audio", Body=audio_bytes)
    print(json.loads(resp["Body"].read().decode("utf-8")))

if __name__ == "__main__":
    main()
