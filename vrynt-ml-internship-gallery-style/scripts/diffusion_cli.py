#!/usr/bin/env python3
import argparse, os, time, json, math
from pathlib import Path
from typing import List
import torch
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    EulerAncestralDiscreteScheduler,
)

SCHEDULERS = {
    "DDPM": DDPMScheduler,
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,   # PLMS-like in diffusers
    "EulerA": EulerAncestralDiscreteScheduler,
}

def build_pipe(model_id: str, sampler: str, device: str = None, eta: float = None, img2img: bool = False):
    Scheduler = SCHEDULERS.get(sampler, DDIMScheduler)
    # Create a default scheduler; diffusers pipelines accept scheduler instances
    if img2img:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device), device

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", action="append", required=True, help="Text prompt (repeatable)")
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--sampler", choices=list(SCHEDULERS.keys()), default="DDIM")
    p.add_argument("--eta", type=float, default=None, help="DDIM eta (0 for deterministic)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--init-image", type=str, default=None, help="Optional path for img2img")
    p.add_argument("--strength", type=float, default=0.6, help="Img2img strength 0..1")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    g = None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)

    img2img = args.init_image is not None
    pipe, device = build_pipe(args.model_id, args.sampler, img2img=img2img, eta=args.eta)

    prompts: List[str] = args.prompt
    n_total = args.num_images
    per_batch = max(1, min(args.batch_size, n_total))
    made = 0
    meta_rows = []

    init_image = None
    if img2img:
        init_image = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))

    while made < n_total:
        bs = min(per_batch, n_total - made)
        t0 = time.time()
        if img2img:
            out = pipe(
                prompt=prompts,
                image=[init_image]*bs,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=g,
                width=args.width,
                height=args.height,
                negative_prompt=args.negative_prompt,
            )
        else:
            out = pipe(
                prompt=prompts,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=g,
                width=args.width,
                height=args.height,
                negative_prompt=args.negative_prompt,
            )
        latency = time.time() - t0

        for img in out.images:
            ts = int(time.time()*1000)
            path = os.path.join(args.outdir, f"gen_{ts}_{made:03d}.png")
            img.save(path)
            meta_rows.append({
                "path": path,
                "prompt": prompts,
                "negative_prompt": args.negative_prompt,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "sampler": args.sampler,
                "seed": args.seed,
                "img2img": img2img,
                "strength": args.strength if img2img else None,
                "latency_s": latency,
                "size": [args.width, args.height],
                "model_id": args.model_id,
            })
            made += 1
            if made >= n_total:
                break

    if args.save_json:
        jpath = os.path.join(args.outdir, "metadata.jsonl")
        with open(jpath, "w") as f:
            for row in meta_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[meta] wrote {jpath}")

    print(f"Saved {made} image(s) to {args.outdir}")

if __name__ == "__main__":
    main()
