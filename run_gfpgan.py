# === 魔法补丁：解决 torchvision 新老版本兼容问题 ===
import sys
import torchvision.transforms.functional
sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional
# ====================================================
import os
import glob
import cv2
from gfpgan import GFPGANer
from tqdm import tqdm

def main():
    input_dir = '/media/witai513/64e0bfd6-4b93-4dd5-9d71-9a1891ecba0d/data/ycy/DiffBIR-main/test'
    output_dir = '/media/witai513/64e0bfd6-4b93-4dd5-9d71-9a1891ecba0d/data/ycy/DiffBIR-main/ntire_gfpgan_drafts'
    
    print("🚀 正在初始化 GFPGAN (首次运行会自动下载权重)...")
    restorer = GFPGANer(
        model_path='/media/witai513/64e0bfd6-4b93-4dd5-9d71-9a1891ecba0d/data/ycy/DiffBIR-main/weights/GFPGANv1.3.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    search_pattern = os.path.join(input_dir, "**", "*.*")
    all_files = glob.glob(search_pattern, recursive=True)
    img_paths = [p for p in all_files if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"✅ 找到 {len(img_paths)} 张测试图，开始提取锐利草稿...")

    for img_path in tqdm(img_paths):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None: continue

        # GFPGAN 处理
        _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        
        # 保持目录结构保存
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # 保存为 PNG
        out_path = os.path.splitext(out_path)[0] + '.png'
        cv2.imwrite(out_path, output)

    print(f"\n🎉 GFPGAN 草稿提取完成！保存在: {output_dir}")

if __name__ == '__main__':
    main()