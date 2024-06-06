import os
import warnings
from deoldify import device
from deoldify.device_id import DeviceId
import torch
from fastai.vision import *
from deoldify.visualize import *
import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image
from pathlib import Path

# Set the device to GPU0
device.set(device=DeviceId.GPU0)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Using 'weights' as positional parameter.*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?torch.nn.utils.weight_norm is deprecated.*?")

# Create required directories
os.makedirs('./models', exist_ok=True)

# Download the image model file if not already present
image_model_url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
image_model_path = "./models/ColorizeArtistic_gen.pth"

if not os.path.exists(image_model_path):
    response = requests.get(image_model_url)
    if response.status_code == 200:
        with open(image_model_path, 'wb') as file:
            file.write(response.content)
        st.success("Image model file downloaded successfully.")
    else:
        st.error("Failed to download the image model file.")

# Download the video model file if not already present
video_model_url = "https://data.deepai.org/deoldify/ColorizeVideo_gen.pth"
video_model_path = "./models/ColorizeVideo_gen.pth"

if not os.path.exists(video_model_path):
    response = requests.get(video_model_url)
    if response.status_code == 200:
        with open(video_model_path, 'wb') as file:
            file.write(response.content)
        st.success("Video model file downloaded successfully.")
    else:
        st.error("Failed to download the video model file.")

# Initialize the colorizers
image_colorizer = get_image_colorizer(artistic=True)
video_colorizer = get_video_colorizer()

# Streamlit UI
st.title("Media Colorizer")
option = st.sidebar.radio("Options", ["Image Colorizer", "Video Colorizer"])

if option == "Image Colorizer":
    uploaded_file = st.file_uploader("Choose a grayscale image", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image_path = Path('./dummy/grayscale_image.jpg')
        image.save(image_path)

        render_factor = st.slider('Render Factor', min_value=10, max_value=40, value=30, step=5)
        watermarked = st.checkbox('Watermarked', value=True)

        if st.button("Colorize Image"):
            result_path = image_colorizer.plot_transformed_image(str(image_path), render_factor=render_factor, compare=True, watermarked=watermarked)
            result_path_str = str(result_path)

            col1, col2 = st.columns(2)
        
            with col1:
                st.image(uploaded_file, caption='Gray-scale Image', use_column_width=True)
        
            with col2:
                st.image(result_path_str, caption='Colorized Image', use_column_width=True)

elif option == "Video Colorizer":
    uploaded_file = st.file_uploader("Choose a grayscale video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            video_path = temp_video_file.name
            temp_video_file.write(uploaded_file.getbuffer())

        render_factor = st.slider('Render Factor', min_value=10, max_value=40, value=16, step=1)
        watermarked = st.checkbox('Watermarked', value=True)

        if st.button("Colorize Video"):
            try:
                result_path = video_colorizer.colorize_from_file_name(str(video_path), render_factor=render_factor, watermarked=watermarked)
                result_path_str = str(result_path)

                st.video(result_path_str, format="video/mp4", start_time=0)
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # col1, col2 = st.columns(2)
        
            # with col1:
            #     st.video(uploaded_file, caption='Gray-scale Video', use_column_width=True)
        
            # with col2:
            #     st.video(result_path_str, caption='Colorized Video', use_column_width=True)


# elif option == "Video Colorizer":
#     uploaded_file = st.file_uploader("Choose a grayscale video", type=["mp4", "avi", "mov"])
    
#     if uploaded_file is not None:
#         video_path = Path('./dummy/grayscale_video.mp4')
#         with open(video_path, 'wb') as f:
#             f.write(uploaded_file.getbuffer())

#         render_factor = st.slider('Render Factor', min_value=10, max_value=40, value=16, step=1)
#         watermarked = st.checkbox('Watermarked', value=True)

#         if st.button("Colorize Video"):
#             try:
#                 result_path = video_colorizer.colorize_from_file_name(str(video_path), render_factor=render_factor, watermarked=watermarked)
#                 result_path_str = str(result_path)

#                 st.video(result_path_str, format="video/mp4", start_time=0)
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")