"""
Script implemented to get a single image from the imagenet dataset, its tag and its idetifier (code)

Authors: Hojin Kang and Tomas Nunez
"""

import glob
import shutil
from keras.preprocessing import image
from PIL import Image
import PIL
import urllib.request

# Path to tags and images
# tags = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/words.txt'
# images ='/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/fall11_urls.txt'
tags = '/home/tomas/Documents/Inteligencia Computacional/tiny-imagenet-200/words.txt'
images = '/home/tomas/Documents/Inteligencia Computacional/Proyecto/fall11_urls.txt'
images_val = '/home/tomas/Documents/Inteligencia Computacional/ILSVRC2012'
tags_val = '/home/tomas/Documents/Inteligencia Computacional/Labels/val.txt'
tags_names = '/home/tomas/Documents/Inteligencia Computacional/Labels/synset_words.txt'

# Function that visualizes a single image and its tag
def single_img(n, height, width):
    """
    Gets a single image and its tag.

    :param      n       : Number of the image to visualize
    :param      height  :size to rescale the height of the image
    :param      width   : size to rescale the width of the image

    :return:    A tuple with the images scaled to size, its tag and the identifier
    """
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

    # Set the user agent
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }

    # Make the request to the page
    request = urllib.request.Request(link, None, headers)
    response = urllib.request.urlopen(request)

    # Get the image from the page and save it
    img = Image.open(response)
    img.save("temp.jpg", "JPEG")

    # Resize image
    size = (height, width)
    img = image.load_img("temp.jpg", target_size=size)

    # Get the name of the class
    clase = identificadores[identifier]

    return img, clase, identifier

def single_img_val(n, height, width):
    """
    Gets a single image from the validation folder, along with its tag and identifier

    :param      n       : Number of the image to visualize
    :param      height  :size to rescale the height of the image
    :param      width   : size to rescale the width of the image

    :return:    A tuple with the images scaled to size, its tag and the identifier
    """
    n_padded = (str(n)).rjust(5, '0')
    imgpath = images_val+'/ILSVRC2012_val_000'+ n_padded +'.JPEG'
    size = (height, width)
    identifier = ''
    clase = ''

    img = image.load_img(imgpath, target_size=size)
    i = 0
    with open(tags_val) as tags:
        for i, line in enumerate(tags):
            if i == n:
                _, tagnumber = line.split(" ", 1)
                with open(tags_names) as names:
                    for j, line2 in enumerate(names):
                        if j == int(tagnumber):
                            identifier, clase = line2.split(" ", 1)
                            break
            elif i > n:
                break
    return img, clase, identifier


