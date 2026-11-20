PEDiT (Phase-Expert Diffusion Transformer)

PEDiT is a diffusion model architecture I'm building for fast, photorealistic image generation on consumer GPUs (targeting 8 GB VRAM and under).

Architecture & Inference
Instead of running a single dense network at every step, PEDiT combines a shared backbone with three phase experts selected by timestep. Each expert handles its own noise range, which cuts active parameters per step, lowers peak VRAM, and avoids task interference between coarse layout and fine detail. I'm using Rectified Flow so the base model generates in 12 steps rather than 24–30. For inference on low-spec hardware (down to 8 GB RAM / 6 GB VRAM), only the active expert stays in VRAM while the others stream asynchronously from disk to RAM to VRAM over background CUDA streams.

150M PoC & Expert Balancing
After the initial 100M generalization test, I trained a 150M-parameter run on 75k image-text pairs (55k human portraits, 20k cats and dogs) at 512x512 (256 latent tokens) using the DC-AE (f=32) autoencoder, batch size 32, and gradient accumulation 3.

Around 60k steps, quality stalled. It wasn't a learning rate issue — checking each expert individually showed that Expert 2 was doing fine while Experts 0 and 1 were lagging behind. I shifted gradient accumulation to 5 with a 40% Expert 0 / 40% Expert 1 / 20% Expert 2 training split and pushed to 85k steps. That broke the plateau: clean facial and animal contours, no cross-species artifacts, and solid prompt adherence for hair color, age, gender, skin tone, and breeds. Main takeaway: experts need identical parameter counts, and the schedule has to weight Experts 0 and 1 heavier.

Comparison grids (60k before rebalance vs 85k after):
assets/60k.jpg
assets/85k.jpg

Moving to FLUX VAE
DC-AE's f=32 compression keeps token counts low, but it physically wipes out micro-textures like skin pores and eyelashes. To hit actual photorealism, I'm switching to the FLUX VAE (f=8 with 2x2 patchification, 4,096 latent tokens at 1024x1024).

Benchmarks on a full 4B INT8 test matrix (matching the target architecture, measured with torch.compile on a desktop RTX 3060, 7.2–7.8 GB peak VRAM):
- DC-AE @ 512x512 (256 tokens): 3.6s for 12 steps
- DC-AE @ 1024x1024 (1,024 tokens): 14s for 12 steps / 18s for 18 steps
- FLUX VAE @ 1024x1024 (4,096 tokens): 20s for 12 steps / 30s for 18 steps

Current Work & Roadmap
Right now I'm running compact ablations to validate the 4,096-token FLUX VAE pipeline, inverted multi-layer text conditioning (feeding Expert 0 late global-semantics layers like Layer 28, and Expert 2 early surface-detail layers like Layer 8), expert balance schedules, progressive resolution scaling (256 -> 512 -> 1024 with 2D RoPE), and a 3x3 MultiScale Depthwise Conv to kill patch seams.

Once the architecture is locked in, the plan is to train and open-source PEDiT Base 1 (4B) with full training, inference, and streaming code, followed by a pruned/distilled Turbo variant (~2–2.5B), an expanded Pro model (~8B), and 4–8 step CFG-less Lightning distillations.
