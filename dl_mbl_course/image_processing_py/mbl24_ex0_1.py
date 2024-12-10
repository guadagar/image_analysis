from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np

'''
DL-MBL-2024 Excercise 0.
Load images, shape, normalize & check data type
'''

img = imread("monuseg-2018/download/images/TCGA-2Z-A9J9-01A-01-TS1.tif")
print(f"Image `img` has type {type(img)}")  # variable type
plt.imshow(img)
mask = imread("monuseg-2018/download/masks/TCGA-2Z-A9J9-01A-01-TS1.tif")
print(f"Mask `mask` has type {type(mask)}")  # variable type
plt.imshow(mask)

#If the image is a grayscale image, then the number of channels is equal to 1, in which case the array can also be of shape (height, width).
#If the image is RGB, then the number of channels is 3. with each channel encoding the red, green and blue components.

print(img.shape) # RGB
print(mask.shape) #grayscale

#plt.show()

#Image data types
#Images can be represented by a variety of data types. The following is a list of the most common datatypes:
#    bool: binary, 0 or 1
#    uint8: unsigned integers, 0 to 255 range
#    float: -1 to 1 or 0 to 1
print("data type: ", img.dtype, mask.dtype)
print("Image min and max: ", img.min(), img.max())
print("Mask min and max: ", mask.min(), mask.max())

#In PyTorch images are represented as (num_channels, height, width).
#Reshape img such that its shape is (num_channels, height, width)
print(f"Before reshaping, image has shape {img.shape}")
img_reshaped = np.transpose(img, (2, 0, 1))
print(f"After reshaping, image has shape {img_reshaped.shape}")

def normalize(img):
    norm_img = img / 255
    return norm_img

print(normalize(img).dtype,normalize(img).min(), normalize(img).max())
