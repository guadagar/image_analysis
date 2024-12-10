import os
from glob import glob
import matplotlib.pyplot as plt
from tifffile import imread
import numpy as np

'''
DL-MBL-2024 Excercise 0.3.
Batching
'''
# `batch` should be a np array with shape (4, 3, 1000, 1000).

img_dir = "monuseg-2018/download/images/"
img_filenames = sorted(glob(os.path.join(img_dir, "*.tif")))

indices = np.random.choice(len(img_filenames), 4, replace=False)
imgs = []
for index in indices:
    imgs.append(np.transpose(imread(img_filenames[index]), (2, 0, 1)))
batch = np.asarray(imgs)
print(f"Batch of images has shape {batch.shape}")
