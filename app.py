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
import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from collections.abc import Sized

# Set the device to GPU0
device.set(device=DeviceId.GPU0)

# # Check if CUDA (GPU) is available
# if not torch.cuda.is_available():
#     st.warning('GPU not available. The process might be slow.')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Using 'weights' as positional parameter.*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?torch.nn.utils.weight_norm is deprecated.*?")

# Create required directories
os.makedirs('./models', exist_ok=True)

# Download the model file if not already present
url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
file_path = "./models/ColorizeArtistic_gen.pth"

if not os.path.exists(file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        st.success("Model File downloaded successfully. You are good to go for colorization!")
    else:
        st.error("Failed to download the model file.")

# Initialize the colorizer
colorizer = get_image_colorizer(artistic=True)

# Streamlit UI
st.title("Media Colorizer")
st.sidebar.radio("Options", ["Image Colorizer"])

uploaded_file = st.file_uploader("Choose a grayscale image", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_path = Path('./dummy/grayscale_image.jpg')
    image.save(image_path)

    render_factor = st.slider('Render Factor', min_value=10, max_value=40, value=30, step=5)
    watermarked = st.checkbox('Watermarked', value=True)

    if st.button("Colorize Image"):
        result_path = colorizer.plot_transformed_image(str(image_path), render_factor=render_factor, compare=True, watermarked=watermarked)
        result_path_str = str(result_path)

        col1, col2 = st.columns(2)
    
        with col1:
            st.image(uploaded_file, caption='Gray-scale Image', use_column_width=True)
    
        with col2:
            st.image(result_path_str, caption='Colorized Image', use_column_width=True)
