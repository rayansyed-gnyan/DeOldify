import os
import warnings
from deoldify import device
from deoldify.device_id import DeviceId
import torch
from fastai.vision import *
from deoldify.visualize import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests

# Set the device to GPU0
device.set(device=DeviceId.GPU0)

# Check if CUDA (GPU) is available
if not torch.cuda.is_available():
    print('GPU not available.')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Using 'weights' as positional parameter.*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?torch.nn.utils.weight_norm is deprecated.*?")

# Create required directories
os.makedirs('./models', exist_ok=True)
os.makedirs('./dummy', exist_ok=True)  # Create the 'dummy' directory

# Download the model file if not already present
url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
file_path = "./models/ColorizeArtistic_gen.pth"

if not os.path.exists(file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("Failed to download the file.")

# Initialize the colorizer
colorizer = get_image_colorizer(artistic=True)

# Set parameters for colorization
image_path = 'gray_scale_image.jpg' 
render_factor = 30
watermarked = True

# Colorize the image from the local path
if os.path.exists(image_path):
    result_path = colorizer.plot_transformed_image(image_path, render_factor=render_factor, compare=True, watermarked=watermarked)
    # Display the image
    img = mpimg.imread(result_path)
    imgplot = plt.imshow(img)
    plt.show()
else:
    print('The specified image path does not exist. Please check the path and try again.')

# Colorize and display images from a local directory with different render factors
for i in range(10, 40, 5):
    colorizer.plot_transformed_image(image_path, render_factor=i, display_render_factor=True, figsize=(8, 8))