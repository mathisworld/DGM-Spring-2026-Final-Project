<div align=center>
  <h1>
  DGM-Spring-2026-Final-Project
  </h1>
  <p>
    <b>UNIST AI518: Deep Generative Models (Spring 2026)</b></a><br>
    Final Project - Face Generation Challenge
  </p>
</div> 

<div align=center>
  <p>
    <b>Seongmin Kim</b></a> (seongmin [at] unist.ac.kr)<br>
    
  </p>
</div>

## Setup

Create a `conda` environment named `dgm`:
```
conda create -n dgm python=3.10 -y
```

Install all required libraries, including PyTorch configured for CUDA 11.8, using the provided `requirements.txt` file.
```
conda activate dgm
pip install -r requirements.txt
```

## Models & Dataset
### Pre-trained Model
This project uses the Latent Diffusion Model (LDM) from `Hugging Face`, pre-trained on the CelebA-HQ dataset. <br>
You can use the following model ID: [CompVis/ldm-celebahq-256]

### Dataset
Download the CelebV-HQ dataset from  <a href="https://drive.google.com/file/d/1lfCZha3ZbxCxY3BUDnxwoyyvyuBd029t/view?usp=drive_link" target="_blank"><b>here</b></a>. <br>
Extract and place the downloaded videos in the target directory:
```
git clone https://github.com/mathisworld/DGM-Spring-2026-Final-Project.git
cd DGM-Spring-2026-Final-Project
mkdir dataset/videos
tar -zxvf celebvhq.tar.gz -C dataset/
```

## Train
### Preprocessing
Before training, the raw dataset must be processed into images and then encoded into latent representations for the Latent Diffusion Model.<br>
Please run the following scripts in order. <br>

#### 1. Extract Frames from Videos (`vid2img.py`)
First, extract frames from the downloaded CelebV-HQ `.mp4` videos. This script will convert the videos into `.jpg` image sequences.
```
python extract_frames.py --input_dir ./dataset/videos --output_dir ./dataset/imgs
```

#### 2. Sample Representative Images (`img_sampler.py`)
To prevent the model from overfitting to highly similar adjacent frames and to optimize training efficiency, this script randomly samples exactly one frame from each video directory and gathers them into a single folder.
```
python img_sampler.py --source_dir ./dataset/imgs --target_dir ./dataset/images_sampled
```

#### 3. Image to Latent Encoding (`get_latent.py`)
Training directly on high-resolution pixel space is computationally expensive. This script encodes the sampled images into the latent space using the pre-trained VAE (CompVis/ldm-celebahq-256) and saves them as PyTorch tensors (.pt). This drastically reduces memory usage and accelerates the training process.
```
python get_latent.py --image_dir ./dataset/images_sampled --latent_dir ./dataset/latents
```

> [!NOTE]
> All scripts are configured with default paths, but you can easily override them using the provided arguments as shown above.