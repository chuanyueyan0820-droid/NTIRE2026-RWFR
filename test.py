import os
import glob
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm

from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config
from diffbir.sampler import SpacedSampler

def load_image(img_path, size=512):
    img = Image.open(img_path).convert("RGB")
    ratio = size / min(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    left = (img.size[0] - size) / 2
    top = (img.size[1] - size) / 2
    img = img.crop((left, top, left + size, top + size))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2 - 1
    return img

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.config)

    print(" 正在加载你 14 万步的 Identity 扩散模型...")
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    cldm.load_pretrained_sd(torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"])
    cldm.load_controlnet_from_ckpt(torch.load(args.ckpt, map_location="cpu"))
    cldm.eval().to(device)

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

    search_pattern = os.path.join(args.draft_input, "**", "*.*")
    all_files = glob.glob(search_pattern, recursive=True)
    img_paths = [p for p in all_files if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    

    with torch.no_grad():
        for imp in tqdm(img_paths, desc="渲染中"):
            clean = load_image(imp).to(device)
            
            prompt = ["high-quality face, realistic skin texture"]
            negative_prompt = ["deformed, distorted, plastic, artificial, exaggerated"]

            cond = cldm.prepare_condition(clean, prompt)
            uncond = cldm.prepare_condition(clean, negative_prompt)
            
            z = sampler.sample(
                model=cldm, device=device, steps=50,
                x_size=(1, 4, 64, 64), cond=cond, uncond=uncond, 
                cfg_scale=1.0
            )

            hq = cldm.vae_decode(z)
            hq = (hq + 1) / 2
            draft_img = (clean + 1) / 2
            hq = hq * 0.55 + draft_img * 0.45 

            hq = hq.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            hq = (hq * 255).astype(np.uint8)

            rel_path = os.path.relpath(imp, args.draft_input)
            out_path = os.path.join(args.output, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            out_path = os.path.splitext(out_path)[0] + ".png"
            Image.fromarray(hq).save(out_path)

    print(f"\n 图像保存在: {args.output}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--draft_input", type=str, required=True, help="GFPGAN生成的草稿文件夹")
    parser.add_argument("--output", type=str, default="ntire_final_submit")
    args = parser.parse_args()
    main(args)
