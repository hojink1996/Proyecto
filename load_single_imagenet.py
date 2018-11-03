'''
Script that implemeted to generate a single image and its tag.

Authors: Hojin Kang and Tomas Nunez
'''

import glob
from PIL import Image
import urllib.request

tags = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/words.txt'
images ='/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/fall11_urls.txt'


# Function that visualizes a single image and its tag
def single_img(n, height, width):
    '''
    Gets a single image and its tag.

    :param      n       : Number of the image to visualize
                height  :size to rescale the height of the image
                width   : size to rescale the width of the image

    :return:    A tuple with the images scaled to size, and its tag
    '''
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

    # Get the image link and identifier
    with open(images) as image:
        for i, line in enumerate(image):
            if i == n:
                identifier = line.split('_')[0]
                link = line.split('http')[1]
                link = 'http' + link
            elif i > n:
                break

    # Get image from URL
    with urllib.request.urlopen(link) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())
    img = Image.open('temp.jpg')

    #Resize image
    size = height, width
    img.thumbnail(size, Image.ANTIALIAS)

    # Get the name of the class
    clase = identificadores[identifier]

    return img, clase
