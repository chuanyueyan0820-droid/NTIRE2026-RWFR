import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
# 确保您已经创建了 id_loss.py 文件
from id_loss import IdentityLoss

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(f"strictly load controlnet weight from pretrained SD")

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in sd.items()}
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    
    # 初始化 Identity Loss 网络
    id_loss_fn = IdentityLoss(device)

    # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    if hasattr(cldm, "module"):
       pure_cldm = cldm.module
    else:
       pure_cldm = cldm
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps with Identity Loss enabled...")

    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_main_process, unit="batch", total=len(loader))
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt = batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                cond = pure_cldm.prepare_condition(clean, prompt)
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(0, noise_aug_timestep, (z_0.shape[0],), device=device),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)

            # --- 修改部分：计算 Total Loss (Diffusion + Identity) ---
            try:
                # 尝试获取 z_0_pred 以便计算 ID Loss
                loss_dict = diffusion.p_losses(cldm, z_0, t, cond_aug, return_pred=True)
                l_simple = loss_dict["loss"]
                z_0_pred = loss_dict["pred"]
                
                # 将预测值和真实值解码到像素空间进行身份对比
                # 注意：VAE 解码非常吃显存，如果报错请减小 batch_size
                img_pred = pure_cldm.vae_decode(z_0_pred)
                img_gt = pure_cldm.vae_decode(z_0)
                
                l_id = id_loss_fn(img_pred, img_gt)
                
                # 总损失权重均衡
                id_weight = 0.2
                loss = l_simple + id_weight * l_id
            except (TypeError, KeyError):
                # 如果代码版本不支持返回 pred，则回退到原始 loss
                loss = diffusion.p_losses(cldm, z_0, t, cond_aug)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/total_step_loss", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # 可视化逻辑
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 4 # 微调阶段减少采样数量以节省显存
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(N, *z_0.shape[1:]),
                        cond={k: v[:N] for k, v in cond.items()},
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        img_save_path = os.path.join(cfg.train.exp_dir, "images")
                        samples = (pure_cldm.vae_decode(z) + 1) / 2
                        grid = make_grid(samples, nrow=2)
                        save_image(grid, os.path.join(img_save_path, f"step_{global_step:07d}_samples.png"))
                        # 同时保存对比图
                        save_image((gt[:N] + 1) / 2, os.path.join(img_save_path, f"step_{global_step:07d}_gt.png"))
                cldm.train()
            
            accelerator.wait_for_everyone()
            if global_step >= max_steps:
                break

        pbar.close()
        epoch += 1

    if accelerator.is_main_process:
        print("done!")
        writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)