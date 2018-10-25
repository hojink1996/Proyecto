'''
Script that loads ALL IMAGES of tiny imagenet and its respective tags.
Also visualizes one of the images with its tag to make sure it works.

Authors: Hojin Kang and Tomas Nunez
'''

import glob
import cv2
import matplotlib.pyplot as plt
import os

# Location of file with tags and images
tags = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/words.txt'
images = glob.glob('/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/train/**/*.JPEG',
                   recursive = True)

# Read lines from the text file with the tags
with open(tags) as tag:
    content = tag.readlines()

# We make a HashMap (dictionary) with the identifiers
identificadores = {}

# Link every directory with it's word
for palabra in content:
    linea = palabra.split('\t')
    codigo = linea[0]
    descriptor = linea[1].strip('\n')
    identificadores[codigo] = descriptor

# Set of the words relating image with it's identifier
image_word = []

# Save images
image_set = []

# Loop over all images
for img in images:
    # Original image
    image = cv2.imread(img)

    # Resize images with INTER_CUBIC interpolation
    res = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    # Append RESIZED image
    image_set.append(res)

    # Get the name of the class
    corresponding_link = os.path.basename(img)
    corresponding_link = corresponding_link.split('_')[0]
    image_word.append(identificadores[corresponding_link])

# Image to visualize
n = 25788

# Visualize an image
imagen =image_set[n]
imgplot = plt.imshow(imagen)
plt.show()
print(image_word[n])