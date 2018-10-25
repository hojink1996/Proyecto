'''
Script that implemeted to generate a single image and its tag.

Authors: Hojin Kang and Tomas Nunez
'''

import glob
import cv2
import os

# Location of file with tags and images
tags = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/words.txt'
images = glob.glob('/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/train/**/*.JPEG',
                   recursive = True)


# Function that visualizes a single image
def single_img(n):

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


    # Get the image
    img = images[n]

    # Original image
    image = cv2.imread(img)

    # Resize images with INTER_CUBIC interpolation
    res = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)


    # Get the name of the class
    corresponding_link = os.path.basename(img)
    corresponding_link = corresponding_link.split('_')[0]
    clase = identificadores[corresponding_link]

    return res, clase
