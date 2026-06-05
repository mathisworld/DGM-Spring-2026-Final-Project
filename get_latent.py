import os
import torch
import argparse
from torchvision import transforms
from PIL import Image
from diffusers import VQModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="이미지들을 VAE를 통해 Latent 텐서로 변환하여 저장합니다.")
    parser.add_argument("--image_dir", type=str, default="/ssdg/spl_seongmin/ws/etc/proj_260514_AI518Final/datasets/dd/datasets_sampled", help="샘플링된 이미지들이 있는 폴더 경로")
    parser.add_argument("--latent_dir", type=str, default="/ssdg/spl_seongmin/ws/etc/proj_260514_AI518Final/datasets/latents", help="변환된 Latent(.pt)를 저장할 폴더 경로")
    
    args = parser.parse_args()

    device = "cuda"
    weight_dtype = torch.bfloat16

    # 1. VAE만 로드
    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae", torch_dtype=weight_dtype).to(device)
    vae.eval()

    # 2. 경로 설정
    os.makedirs(args.latent_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # [0, 1] -> [-1, 1] 변환
    ])

    images = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg'))]

    # 3. 모든 이미지를 순회하며 Latent 추출 및 저장
    print(f"총 {len(images)}장의 이미지를 Latent로 변환합니다... (입력: {args.image_dir})")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
        for img_name in tqdm(images):
            img_path = os.path.join(args.image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device, dtype=weight_dtype)
            
            # VAE 압축 및 스케일링 (0.18215 곱하기까지 미리 해둠!)
            # latent = vae.encode(img_tensor).latents * 0.18215
            latent = vae.encode(img_tensor).latents
            
            # .pt 파일로 저장 (예: image_001.jpg -> image_001.pt)
            save_name = os.path.splitext(img_name)[0] + ".pt"
            torch.save(latent.squeeze(0).cpu(), os.path.join(args.latent_dir, save_name))

    print("✅ 모든 Latent 추출 완료!")
    print(f"저장 경로: {args.latent_dir}")

if __name__ == "__main__":
    main()