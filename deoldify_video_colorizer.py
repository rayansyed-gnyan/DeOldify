import deoldify
import os
import warnings
import requests
import torch
import ffmpeg
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *
from pathlib import Path
from PIL import Image

def download_model(url, dest):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print(f"Failed to download file from {url}")

# Clone DeOldify repo if not already present
if not os.path.exists('DeOldify'):
    os.system('git clone https://github.com/jantic/DeOldify.git DeOldify')
os.chdir('DeOldify')

# Ensure the dummy directory and dummy file exists
dummy_path = Path('./dummy/')
dummy_path.mkdir(parents=True, exist_ok=True)
dummy_file = dummy_path / 'dummy_image.jpg'
if not dummy_file.exists():
    Image.new('RGB', (1, 1)).save(dummy_file)  # Create a 1x1 black image

# Set device to GPU
device.set(device=DeviceId.GPU0)

# Install required packages
os.system('pip install -r requirements-colab.txt')

# Set configuration for PyTorch and warnings
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

# Create models directory if not exists
if not os.path.exists('models'):
    os.mkdir('models')

# Download pre-trained model
model_url = 'https://data.deepai.org/deoldify/ColorizeVideo_gen.pth'
model_path = './models/ColorizeVideo_gen.pth'
if not os.path.exists(model_path):
    download_model(model_url, model_path)

# Initialize video colorizer
colorizer = get_video_colorizer()

# Parameters for video colorization
input_video_path = 'grayscale_video3.mp4'

print(f"Current working directory: {os.getcwd()}")
full_video_path = os.path.abspath(input_video_path)
print(f"Full path to input video: {full_video_path}")
print("System PATH:", os.environ["PATH"])

render_factor = 16
watermarked = True

if os.path.exists(full_video_path):
    video_path = colorizer.colorize_from_file_name(full_video_path, render_factor, watermarked=watermarked)
    print(f'Video saved at: {video_path}')
else:
    print(f'Input video file not found at {full_video_path}. Provide a valid video file path and try again.')
