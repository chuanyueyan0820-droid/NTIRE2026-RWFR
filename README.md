## Prepare the environment

```
pip install -r requirements.txt
```

## How to test the model?

1.  Download our model from `model_zoo\team04_PRIDE-Face\model.txt`，put it in `model_zoo\team04_PRIDE-Face`

2.  Download pretrained model `SD2.1` , put it in `pretrained_models/stable-diffusion-2-1-base`

3.  Modify `args.pretrained_model_path` and `args.pretrained_path` in `models\team08_drre\io.py`, please **DON'T** modify other options for best  performance 

   
## 🛠️ Inference
To reproduce our final submission (which achieved highly natural textures), run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --config configs/train/train_stage2.yaml \
    --ckpt <our model> \
    --draft_input <path_to_gfpgan_drafts> \
    --output <path to your save dir># NTIRE2026-RWFR
