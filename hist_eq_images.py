#!/usr/bin/env python3
import sys
import os
import glob
import cv2

"""
histogram equalization with OpenCV, it enhances low contrast images. Eq images have a ~ linear cum dist func
GCG
09.24
"""

def saveImage(img, ofn, qual=None, comp=None):
    if qual is None:
        ext = os.path.splitext(ofn)[-1]
        if (ext == '.tif') or (ext == '.tiff') or (ext == '.TIF') or (ext == '.TIFF'):
            if comp != None:
                # code 1 means uncompressed tif
                # code 5 means LZW compressed tif
                cv2.imwrite(ofn, img, (cv2.IMWRITE_TIFF_COMPRESSION, comp))
            else:
                # Use default
                cv2.imwrite(ofn, img)
        else:
            cv2.imwrite(ofn, img)
    else:
        cv2.imwrite(ofn, img, (cv2.IMWRITE_JPEG_QUALITY, qual))


folder_path = './original_images/'
outdir = './mod_images'

for image_file in glob.glob(os.path.join(folder_path,'*.tif')):
  print("Processing: %s" % (image_file))
  img = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl1 = clahe.apply(img)

  img_rev = cl1#img_mod
  basename = os.path.split(image_file)[-1]
  outname = os.path.join(outdir,basename)
  print(image_file,basename, outname)

  saveImage(img_rev, outname, comp=1)

#ax=plt.figure(figsize=(20,20))

# Plotting the original image
##plt.subplot(221)
#plt.title('Original')
#plt.imshow(img, cmap=plt.cm.gray)
##plt.hist(img.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])

#plt.subplot(222)
#plt.imshow(img_rev, cmap=plt.cm.gray)
#plt.hist(img_rev.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.show()
