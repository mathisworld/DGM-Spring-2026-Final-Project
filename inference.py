import os
import argparse

# 🌟 핵심 1: PyTorch가 실행되기 전에 CUDA 환경 변수를 강제로 고정합니다.
# 이 설정이 없으면 결정론적(Deterministic) 알고리즘이 아예 작동하지 않습니다.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)

import torch
import random
import numpy as np
import torchvision.utils as vutils
from diffusers import VQModel, UNet2DModel, DDIMScheduler 
from tqdm import tqdm

# 🌟 핵심 2: 현존하는 모든 시드 및 연산 방식을 고정하는 궁극의 함수
def set_absolute_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 환경 방어
    
    # cuDNN이 계산 속도를 높이려고 맘대로 알고리즘을 바꾸는 것을 차단
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 내부 연산 알고리즘을 무조건 결정론적인 것만 쓰도록 강제
    # (일부 연산에서 속도가 조금 느려질 수 있습니다)
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    parser = argparse.ArgumentParser(description="학습된 LDM 모델을 사용하여 1000장의 이미지를 생성합니다.")
    parser.add_argument("--local_model_path", type=str, default="/ssdg/spl_seongmin/ws/etc/proj_260514_AI518Final/saved_models/t3", help="학습된 UNet 가중치가 저장된 폴더 경로")
    parser.add_argument("--output_dir", type=str, default="/ssdg/spl_seongmin/ws/etc/proj_260514_AI518Final/inference/t3_test2", help="생성된 이미지를 저장할 폴더 경로")
    
    args = parser.parse_args()

    # 베이스 시드 적용!
    set_absolute_deterministic_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.bfloat16

    print("1. 모델 및 스케줄러를 불러옵니다...")
    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae", torch_dtype=weight_dtype).to(device)
    vae.eval()

    unet = UNet2DModel.from_pretrained(args.local_model_path, torch_dtype=weight_dtype).to(device)
    unet.eval()

    # DDIM 스케줄러 사용
    noise_scheduler = DDIMScheduler.from_pretrained("CompVis/ldm-celebahq-256", subfolder="scheduler")

    print("2. 1000장 생성을 시작합니다...")
    total_images = 1000
    num_inference_steps = 50 

    latent_shape = (1, 3, 64, 64) 

    os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
        for i in tqdm(range(total_images), desc="Generating Images"):
            
            # 각 이미지마다 고유하면서도 재현 가능한 시드 부여
            current_seed = 42 + i 
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            x_t = torch.randn(latent_shape, generator=generator, device=device, dtype=weight_dtype)
            
            for t in noise_scheduler.timesteps:
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = unet(x_t, t_batch).sample
                x_t = noise_scheduler.step(noise_pred, t, x_t, generator=generator).prev_sample
                
            decoded_images = vae.decode(x_t).sample
            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
            
            # Leaderboard 제출 규격에 맞춘 파일명 (img_0000.jpg ~ img_0999.jpg)
            output_filename = os.path.join(args.output_dir, f"img_{i:04d}.jpg")
            vutils.save_image(decoded_images.float(), output_filename)

    print(f"\n✅ 1000장 생성 완료! '{args.output_dir}' 폴더를 확인해 보세요.")

if __name__ == "__main__":
    main()