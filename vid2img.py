import cv2
import os
import glob
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="비디오 파일에서 프레임을 추출합니다.")
    parser.add_argument("--input_dir", type=str, default="/home/seongmin/Downloads/35666", help="mp4 파일이 있는 입력 폴더 경로")
    parser.add_argument("--output_dir", type=str, default="/home/seongmin/Downloads/datasets", help="프레임이 저장될 출력 폴더 경로")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_paths = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    print(f"총 {len(video_paths)}개의 영상을 변환합니다... (입력: {args.input_dir})")

    for vid_path in tqdm(video_paths):
        vid_name = os.path.basename(vid_path).split('.')[0]
        vid_out_dir = os.path.join(args.output_dir, vid_name)
        os.makedirs(vid_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                save_path = os.path.join(vid_out_dir, f"{frame_idx:05d}.jpg")
                cv2.imwrite(save_path, frame)
                
            frame_idx += 1
            
        cap.release()

    print("데이터셋 변환 완료!")

if __name__ == "__main__":
    main()