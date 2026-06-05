import torch
import torch.nn.functional as F
import torchvision.utils as vutils
# 💡 핵심: DDPMScheduler 추가
from diffusers import VQModel, UNet2DModel, DDPMScheduler 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import random
import numpy as np
import argparse

# ---------------------------------------------------------
# 1. 시드(Seed) 고정
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path)
        return latent

def main():
    # =========================================================
    # Argument Parser 설정
    # =========================================================
    parser = argparse.ArgumentParser(description="LDM 파인튜닝 학습 스크립트")
    parser.add_argument("--latent_dir", type=str, default="/ssdg/spl_seongmin/ws/etc/proj_260514_AI518Final/datasets/latents", help="학습에 사용할 Latent(.pt) 파일들이 있는 폴더 경로")
    parser.add_argument("--output_dir", type=str, default="/ssdg/spl_seongmin/ws/etc/proj_260514_AI518Final/saved_models/t10", help="학습된 모델과 시각화 이미지를 저장할 폴더 경로")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device with PyTorch {torch.__version__}")

    weight_dtype = torch.bfloat16

    # ---------------------------------------------------------
    # 2. 모델 및 스케줄러 로드
    # ---------------------------------------------------------
    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae", torch_dtype=weight_dtype).to(device)
    unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet", torch_dtype=weight_dtype).to(device)

    # 💡 모델이 원래 학습되었던 노이즈 스케줄러를 그대로 가져옵니다.
    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/ldm-celebahq-256", subfolder="scheduler")

    unet.enable_gradient_checkpointing()

    vae.eval()
    vae.requires_grad_(False)
    unet.train()

    # 학습률은 5e-6으로 유지하여 기존 지식을 잃지 않게 보호합니다.
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6) 

    # ---------------------------------------------------------
    # 3. 데이터 로더 (Latent)
    # ---------------------------------------------------------
    latent_dataset = LatentDataset(args.latent_dir)
    dataloader = DataLoader(latent_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    # ---------------------------------------------------------
    # 4. 고정된 노이즈(Fixed Noise) 생성 (시각화용)
    # ---------------------------------------------------------
    print("Generating fixed noise for visualization...")
    dummy_latent = next(iter(dataloader)).to(device)
    latent_shape = (5, *dummy_latent.shape[1:]) 
    fixed_noise = torch.randn(latent_shape, device=device, dtype=weight_dtype)

    # 5. 저장 경로 설정
    num_epochs = 20 
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 6. Epoch 단위 반복 학습
    # ---------------------------------------------------------
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training")
        
        # ==========================
        # [Train Phase - LDM 방식]
        # ==========================
        for batch_latents in progress_bar: 
            x_1 = batch_latents.to(device, dtype=weight_dtype)
            batch_size = x_1.shape[0]

            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                # 1. 원본 Latent에 더할 순수 노이즈 생성
                noise = torch.randn_like(x_1) 
                
                # 2. 배치 내 각 이미지마다 0 ~ 1000 사이의 랜덤한 시간(timestep) 스텝 뽑기
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
                ).long()
                
                # 3. 스케줄러를 이용해 x_1에 해당 timestep만큼 노이즈 섞기 (Forward Diffusion)
                noisy_latents = noise_scheduler.add_noise(x_1, noise, timesteps)
                
                # 4. 모델 예측: "지금 섞여 있는 노이즈(noise)가 무엇인지 맞춰봐!"
                noise_pred = unet(noisy_latents, timesteps).sample
                
                # 5. Loss 계산: 정답 노이즈 vs 예측된 노이즈
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # ==========================
        # [Validation Phase - LDM 방식]
        # ==========================
        print("Generating visualization images...")
        unet.eval() 
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
            x_t = fixed_noise.clone() 
            
            # 💡 LDM(DDPM)은 Flow Matching보다 스텝을 더 많이 밟아야 형태가 나옵니다.
            num_inference_steps = 50 
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            
            # 스케줄러의 타임스텝을 역순(1000 -> 0)으로 순회하며 노이즈 깎기
            for t in noise_scheduler.timesteps:
                # 5개 이미지에 동일한 시간 스텝 적용
                t_batch = torch.full((5,), t, device=device, dtype=torch.long)
                
                # 현재 노이즈 예측
                noise_pred = unet(x_t, t_batch).sample
                
                # 스케줄러를 통해 노이즈를 한 스텝 벗겨낸 이전 상태(x_{t-1}) 계산
                x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample
                
            # VAE 디코딩
            # x_t = x_t / 0.18215
            decoded_images = vae.decode(x_t).sample
            
        decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
        
        vis_path = os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.png")
        vutils.save_image(decoded_images.float(), vis_path, nrow=5)
        print(f"✅ 시각화 이미지 저장 완료: {vis_path}")
        
        unet.train() 

    # 7. 학습 완료 후 가중치 저장
    print(f"\n🎉 학습이 완료되었습니다. 모델을 {args.output_dir} 에 저장합니다...")
    unet.save_pretrained(args.output_dir)
    print("저장 완료!")

if __name__ == "__main__":
    main()