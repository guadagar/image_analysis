import os
from glob import glob
import matplotlib.pyplot as plt
from tifffile import imread
import numpy as np
from tqdm import tqdm

'''
DL-MBL-2024 Excercise 0.4
Convolution, image with a kernel.
'''

def conv2d(img, kernel):
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 != 0

    h, w = img.shape[0], img.shape[1]  # Starting size of image
    d_k = kernel.shape[0]  # Size of kernel

    h_new = h - d_k + 1
    w_new = w - d_k + 1
    output = np.zeros((h_new, w_new))

    for i in tqdm(range(output.shape[0]), desc="Processing rows", position=0, leave=True):
        for j in tqdm(range(output.shape[1]),desc="Processing columns", position=0, leave=False):
            output[i, j] = np.sum(img[i:i + d_k, j:j + d_k] * kernel)
    return output

img_dir = "monuseg-2018/download/images/"
img_filenames = sorted(glob(os.path.join(img_dir, "*.tif")))

idx = 0#np.random.randint(len(img_filenames))
img = imread(img_filenames[idx])
identity = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])
#solver filter
#flter = np.array([[1, 2, 1],
#                  [0, 0, 0],
#                  [-1, -2, -1]])

#ridges filter
flter = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]])

 #Gaussian blur filter
#flter = np.array([[1, 2, 1],
#                [2, 4, 2],
#                [1, 2, 1]])
# Let's take a 256x256 center crop of the image for better visualization of the effect of the convolution
new_im = conv2d(img[128:384, 128:384, 0], flter)
# Lets print the original image and the convolved image
print(img[128:384, 128:384, 0].shape)
print(new_im.shape)

#Lets visualize the original image and the convolved image and the filter
plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(img[128:384, 128:384, 0])
plt.title("Original Image")
plt.subplot(132)
plt.imshow(identity)
plt.title("Kernel")
plt.subplot(133)
plt.imshow(new_im)
plt.title("Convolved Image")
plt.tight_layout()
plt.show()
