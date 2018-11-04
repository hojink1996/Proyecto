'''
Script that implemeted to generate a single image and its tag.

Authors: Hojin Kang and Tomas Nunez
'''

import glob
import shutil
from keras.preprocessing import image
from PIL import Image
import PIL
import urllib.request

# Tags and images
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
    with open(images) as image_opener:
        for i, line in enumerate(image_opener):
            if i == n:
                identifier = line.split('_')[0]
                link = line.split('http')[1]
                link = 'http' + link
                print(link)
            elif i > n:
                break

    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(link, None, headers)
    response = urllib.request.urlopen(request)

    img = Image.open(response)
    img.save("temp.jpg", "JPEG")

    # Resize image
    size = (height, width)
    img = image.load_img("temp.jpg", target_size=size)

    # Get the name of the class
    clase = identificadores[identifier]

    return img, clase, identifier
