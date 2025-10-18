# inference.py
# Minimal inference handler if using a custom container (for reference).
# For HF DLC, this is not required, but kept for future custom models.

import json, io
import soundfile as sf
from transformers import AutoModelForCTC, AutoProcessor
import torch

model = None
processor = None

def model_fn(model_dir):
    global model, processor
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type.startswith("application/x-audio"):
        data, sr = sf.read(io.BytesIO(request_body))
        return {"waveform": data, "sr": sr}
    raise ValueError("Unsupported content type")

@torch.inference_mode()
def predict_fn(inputs, model):
    inputs_pt = processor(inputs["waveform"], sampling_rate=inputs["sr"], return_tensors="pt")
    logits = model(**inputs_pt).logits
    ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(ids)[0]
    return {"text": text}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
