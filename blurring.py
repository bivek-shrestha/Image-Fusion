import cv2 as cv
import os
import matplotlib.pyplot as plt

import numpy as np
import random

# Specify the path to the folder containing your images
folder_path = 'testdataset'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Loop through each image file
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, image_file)
    output_image_path = os.path.join(output_folder, image_file)

    # Read the image using OpenCV
    img = cv.imread(image_path)
    
    # Apply median blur to the entire image
    larger_kernel_size =17
    blurred_img = cv.medianBlur(img, larger_kernel_size)
    
    cv.imwrite(output_image_path, blurred_img)

# Release resources
cv.destroyAllWindows()