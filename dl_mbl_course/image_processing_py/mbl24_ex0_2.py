import os
from glob import glob
import matplotlib.pyplot as plt
from tifffile import imread
import numpy as np

'''
DL-MBL-2024 Excercise 0.2.
Load all the files, visualize, cropped images, downsampling, flipping
'''

img_dir = "monuseg-2018/download/images/"
img_filenames = sorted(glob(os.path.join(img_dir, "*.tif")))

print(f"Found:")

for img_filename in img_filenames:
    print(f"{img_filename}")

mask_dir = "monuseg-2018/download/masks/"
mask_filenames = sorted(glob(os.path.join(mask_dir, "*.tif")))

for mask_filename in mask_filenames:
    print(f"{mask_filename}")

def visualize(im1, im2):
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)
    plt.tight_layout()

idx = 8 #change this value to visualize a different image and mask to explore the dataset
visualize(imread(img_filenames[idx]), imread(mask_filenames[idx]))

#cropping Images
idx = np.random.randint(len(img_filenames))
img = imread(img_filenames[idx])
cropped_img = img[0:500, 0:500, :]
visualize(img, cropped_img)

#Visualize the bottom left (third) quadrant of a random image
idx = np.random.randint(len(img_filenames))
img = imread(img_filenames[idx])
cropped_img = img[500:, 0:500, :]
visualize(img, cropped_img)

# downsampling
idx = np.random.randint(len(img_filenames))
img = imread(img_filenames[idx])

factor = 4
downsampled_img = img[::factor, ::factor] # here we are selecting every 'factor' pixel in the height and width dimension
print(f"Original image shape: {img.shape}")
print(f"Downsampled image shape: {downsampled_img.shape}")

# Let's visualize the original image and the downsampled image side by side
visualize(img, downsampled_img)

#Flipping
idx = np.random.randint(len(img_filenames))
img = imread(img_filenames[idx])
# Here the image dimensions are (height, width, num_channels), ::-1 means reverse the order of the elements in the array on the height axis
vflipped_img = img[::-1, :, :]
visualize(img, vflipped_img)

#horizontally flipped image
idx = np.random.randint(len(img_filenames))
img = imread(img_filenames[idx])
hflipped_img = img[:, ::-1, :]
visualize(img, hflipped_img)
plt.show()
