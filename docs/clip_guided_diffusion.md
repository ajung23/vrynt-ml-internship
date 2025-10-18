# CLIP-guided diffusion — engineering notes

**Goal.** Summarize the CLIP-conditioned diffusion setup and expose a practical, reproducible interface for demos and quick experiments.

## What it is
- A denoising diffusion network trained to reverse a gradual noising process and generate images from text/image conditions.
- Conditioning: 512‑D CLIP text/image embeddings guide sampling.
- Training note: model referenced in legacy notes is ~602M params, trained ~3.1M steps and fine‑tuned with classifier‑free guidance. (Historical context — not required to run the demo).

## Parameters (mapped to our CLI)
| Parameter (legacy) | Meaning | Our CLI flag |
| --- | --- | --- |
| `--prompts` | One or more text prompts; optional weights after a colon | `--prompt` (repeatable) |
| `--batch-size` | Images per batch | `--batch-size` |
| `--checkpoint` | Model id or local path to weights | `--model-id` |
| `--clip-guidance-scale` | Strength of guidance | `--guidance-scale` |
| `--cutn` / `--cut-pow` | Random crop heuristics for CLIP guidance | *(n/a in diffusers; not needed)* |
| `--device` | cuda / cpu | auto |
| `--eta` | Determinism for DDIM (0..1) | `--eta` (DDIM only) |
| `--images` | Additional image prompts or init image | `--init-image` (img2img) |
| `--init` | Init image path | `--init-image` |
| `--method` | Sampler (DDPM, DDIM, PLMS, PRK/PIE variants) | `--sampler` (DDPM, DDIM, PNDM, EulerA) |
| `--model` | Model family | `--model-id` |
| `-n` | Number of samples | `--num-images` |
| `--seed` | RNG seed | `--seed` |
| `--starting-timestep` | Img2img start strength (0..1) | `--strength` |
| `--size` | Output resolution | `--width`, `--height` |
| `--steps` | Diffusion steps | `--steps` |

> Notes
> - Some low‑level CLIP‑crop parameters are library‑specific. In this repo, we use Hugging Face **diffusers** with standard guidance and schedulers; behavior aligns with modern practice and is reproducible.

## Quickstart
```bash
# Text -> image (CPU or GPU)
python scripts/diffusion_cli.py --prompt "clean studio product mockup" --steps 30 --guidance-scale 7.5

# Repeatable generation
python scripts/diffusion_cli.py --prompt "minimalist icon, vector style" --seed 42 --num-images 4

# Img2Img with strength
python scripts/diffusion_cli.py --prompt "oil painting style" --init-image path/to/input.png --strength 0.6

# Change sampler
python scripts/diffusion_cli.py --prompt "soft lighting, editorial" --sampler DDIM --eta 0.0
```

## Outputs
- Images saved to `outputs/` with timestamped filenames.
- Optional JSON sidecar with prompt, seed, steps, guidance, sampler, and wall‑clock latency for each image.

## Repro tips
- Fix `--seed`, `--steps`, and `--guidance-scale` for demo parity.
- Keep sizes as multiples of 8; 512×512 is a solid default.
- For img2img, use `--strength` in 0.3–0.7 for noticeable but stable edits.

