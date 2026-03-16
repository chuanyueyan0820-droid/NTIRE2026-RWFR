## Prepare the environment

```
pip install -r requirements.txt
pip install gfpgan basicsr  
```

## How to test the model?

1.  Download our model from `model_zoo\team04_PRIDE-Face\model.txt`，put it in `model_zoo\team04_PRIDE-Face`
**⚠️ Please Note:** There are two different key models included that you need to be aware of:
* `v2-1_512-ema-pruned.ckpt`: The SD2.1 base model. Please ensure its path is correctly configured in our `train_stage2.yaml` (or `stage2.yaml`) under the `train:` section (`sd_path: model_zoo/team04_PRIDE-face/v2-1_512-ema-pruned.ckpt`).
* `04.pt`: Our core 140k-step fine-tuned pre-trained model. This will be explicitly loaded via the `--ckpt` argument when running the test command.
  
2.  Stage 1: Extract Structural Prior (GFPGAN)
To avoid the over-smoothing issue caused by severe degradation, we first extract a sharp facial structural prior. Run the pre-processing script to generate the intermediate drafts:
```
python run_gfpgan.py --input_dir [path to test data dir] --output_dir [path_to_gfpgan_drafts]
```
   Stage 2: Generate High-Fidelity Details (DiffBIR Fine-tuned with ID Loss)
Feed the intermediate drafts into our identity-preserving diffusion model to get the final results:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --config configs/train/train_stage2.yaml \
    --ckpt model_zoo/team04_PRIDE-face/04.pt \
    --draft_input [path_to_gfpgan_drafts] \
    --output [path to your save dir]
```
