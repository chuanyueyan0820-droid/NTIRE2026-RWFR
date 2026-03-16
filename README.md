## Prepare the environment

```
pip install -r requirements.txt
pip install gfpgan basicsr  # Required for our Stage 1 structural prior extraction
```

## How to test the model?

1.  Download our model from `model_zoo\team04_PRIDE-Face\model.txt`，put it in `model_zoo\team04_PRIDE-Face`

2.  Stage 1: Extract Structural Prior (GFPGAN)
To avoid the over-smoothing issue caused by severe degradation, we first extract a sharp facial structural prior. Run the pre-processing script to generate the intermediate drafts:
```
python run_gfpgan.py --input_dir [path to test data dir] --output_dir [path_to_gfpgan_drafts]
```
3.  Stage 2: Generate High-Fidelity Details (DiffBIR Fine-tuned with ID Loss)
Feed the intermediate drafts into our identity-preserving diffusion model to get the final results:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --config configs/train/train_stage2.yaml \
    --ckpt <our model> \
    --draft_input <path_to_gfpgan_drafts> \
    --output <path to your save dir>
```
