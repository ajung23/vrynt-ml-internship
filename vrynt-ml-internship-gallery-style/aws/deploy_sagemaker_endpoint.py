# deploy_sagemaker_endpoint.py
import argparse, json, time, os
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import Session

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--endpoint-name", default="vrynt-stt-demo")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    sess = Session(boto3.session.Session(region_name=args.region))
    role = os.environ.get("SAGEMAKER_ROLE_ARN") or sess.get_caller_identity_arn()

    env = {
        "HF_TASK": "automatic-speech-recognition",
        "HF_MODEL_ID": args.model_id,
    }

    model = HuggingFaceModel(
        env=env,
        role=role,
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        sagemaker_session=sess,
    )

    print(f"Deploying endpoint {args.endpoint_name} with model {args.model_id} ...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )
    print("Deployed. Test invoke with boto3 or the Streamlit app.")

if __name__ == "__main__":
    main()
