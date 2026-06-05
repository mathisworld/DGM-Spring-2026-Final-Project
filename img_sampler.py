import os
import random
import shutil
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="각 비디오 폴더에서 랜덤으로 이미지 1장씩을 샘플링합니다.")
    parser.add_argument("--source_dir", type=str, default="/home/seongmin/ws/etc/proj_260514_AI518Final/datasets", help="원본 데이터셋(폴더들)이 있는 경로")
    parser.add_argument("--target_dir", type=str, default="/home/seongmin/ws/etc/proj_260514_AI518Final/datasets_sampled", help="추출된 이미지가 저장될 새로운 폴더 경로")
    
    args = parser.parse_args()

    # 타겟 폴더가 없으면 생성
    os.makedirs(args.target_dir, exist_ok=True)

    # 2. 원본 폴더 내의 모든 하위 폴더 목록 가져오기
    subdirs = [os.path.join(args.source_dir, d) for d in os.listdir(args.source_dir) 
               if os.path.isdir(os.path.join(args.source_dir, d))]

    print(f"총 {len(subdirs)}개의 비디오 폴더를 발견했습니다. (입력: {args.source_dir})")
    print(f"각 폴더에서 랜덤으로 1장씩 추출하여 '{args.target_dir}'로 복사합니다...")

    count = 0
    valid_exts = ('.png', '.jpg', '.jpeg')

    # 3. 각 폴더를 돌면서 딱 1장만 뽑아오기
    for subdir in tqdm(subdirs, desc="Sampling Images"):
        # 해당 폴더 안의 이미지 파일 목록 가져오기
        images = [f for f in os.listdir(subdir) if f.lower().endswith(valid_exts)]
        
        if not images:
            continue # 이미지가 없는 폴더는 패스
            
        # 랜덤으로 한 장 선택 (비디오 프레임 중 무작위 1장)
        selected_image = random.choice(images)
        src_path = os.path.join(subdir, selected_image)
        
        # 🌟 이름 충돌 방지: "폴더명_기존파일명.jpg" 형태로 저장
        # 예: video_001 폴더의 frame_05.jpg -> video_001_frame_05.jpg
        folder_name = os.path.basename(subdir)
        new_filename = f"{folder_name}_{selected_image}"
        tgt_path = os.path.join(args.target_dir, new_filename)
        
        # 이미지 복사 (이동이 아니므로 원본은 안전하게 보존됨)
        shutil.copy2(src_path, tgt_path)
        count += 1

    print("\n🎉 샘플링 완료!")
    print(f"총 {count}장의 이미지가 성공적으로 추출되었습니다.")
    print(f"새로운 데이터셋 경로: {args.target_dir}")

if __name__ == "__main__":
    main()