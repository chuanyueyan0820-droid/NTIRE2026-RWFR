# NTIRE 2026 Real-World Face Restoration Challenge
Team: 0820chuanyueyan

This repository contains the official inference code for our submission to the NTIRE 2026 Real-World Face Restoration Challenge.

## 🚀 Method Overview
Our method utilizes a robust Two-Stage Prior-Guided Diffusion Architecture** to balance perceptual naturalness and structural fidelity:
1. Stage 1 (Coarse Restoration): We employ GFPGAN to extract sharp facial structures and remove severe degradations from the LQ inputs.
2. Stage 2 (Fine Restoration): A Latent Diffusion Model (based on DiffBIR) is conditioned on the Stage 1 outputs. To avoid the "plastic" artifact and metric-oriented over-sharpening, we strictly lock the `cfg_scale` to 1.5, generating highly realistic human skin textures with natural breathability. We also incorporate an Identity Loss to preserve the subject's original facial characteristics.

## 🛠️ Inference
To reproduce our final submission (which achieved highly natural textures), run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_ntire_v2.py \
    --config configs/train/train_stage2.yaml \
    --ckpt <path_to_your_140k_checkpoint> \
    --draft_input <path_to_gfpgan_drafts> \
    --output ntire_final_submit# NTIRE2026-RWFR
