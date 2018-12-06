"""
Script implemented to get a single image from the imagenet dataset, its tag and its idetifier (code)

Authors: Hojin Kang and Tomas Nunez
"""

from keras.preprocessing import image
from PIL import Image
import urllib.request
import numpy as np

# Path to tags and images
tags = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/words.txt'
images ='/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/fall11_urls.txt'
# tags = '/home/tomas/Documents/Inteligencia Computacional/tiny-imagenet-200/words.txt'
# images = '/home/tomas/Documents/Inteligencia Computacional/Proyecto/fall11_urls.txt'
images_val = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/ILSVRC2012Val'
tags_val = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/val.txt'
tags_names = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/synset_words.txt'
adversarial = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/Adversarios/'


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
            if i == n-1:
                _, tagnumber = line.split(" ", 1)
                with open(tags_names) as names:
                    for j, line2 in enumerate(names):
                        if j == int(tagnumber):
                            identifier, clase = line2.split(" ", 1)
                            break
            elif i > n:
                break
    return img, clase, identifier


def n_images_validation(n_ini, n_fin, height, width):
    """
    Gets n images from the validation folder, along with its tag and identifier

    :param      n_ini   : Value of first image
    :param      n_fin   : Value of las image
    :param      height  : size to rescale the height of the image
    :param      width   : size to rescale the width of the image

    :return:    A tuple with the list of images scaled to size, its tag and identifiers
    """
    imgpaths = []
    for i in np.arange(n_ini, n_fin):
        n_padded = (str(i + 1)).rjust(5, '0')
        imgpaths.append(images_val+'/ILSVRC2012_val_000'+ n_padded +'.JPEG')
    size = (height, width)
    identifiers = []
    clases = []
    images = []

    for i in range(len(imgpaths)):
        img = image.load_img(imgpaths[i], target_size=size)
        images.append(img)

    with open(tags_val) as tags:
        for i, line in enumerate(tags):
            if n_ini - 1< i < n_fin:
                _, tagnumber = line.split(" ", 1)
                with open(tags_names) as names:
                    for j, line2 in enumerate(names):
                        if j == int(tagnumber):
                            identifier, clase = line2.split(" ", 1)
                            identifiers.append(identifier)
                            clases.append(clase)
                            break
            elif i > n_fin:
                break
    return images, clases, identifiers


def n_images_adversarial(n_ini, n_fin, height, width):
    """
    Gets n images from the adversarial examples

    :param      n_ini   : Value of first image
    :param      n_fin   : Value of las image
    :param      height  : size to rescale the height of the image
    :param      width   : size to rescale the width of the image

    :return:    A list with the adversarial examples
    """
    imgpaths = []
    for i in np.arange(n_ini, n_fin):
        imgpaths.append(adversarial + str(i + 1) +'.JPEG')
    size = (height, width)
    images = []

    for i in range(len(imgpaths)):
        img = image.load_img(imgpaths[i], target_size=size)
        images.append(img)

    return images


def n_arrays_adversarial(n_ini, n_fin, height, width):
    """
    Gets n images from the adversarial examples

    :param      n_ini   : Value of first image
    :param      n_fin   : Value of las image
    :param      height  : size to rescale the height of the image
    :param      width   : size to rescale the width of the image

    :return:    A list with adversarial examples in array form
    """
    imgpaths = []
    for i in np.arange(n_ini, n_fin):
        imgpaths.append(adversarial + str(i + 1) +'.JPEG')
    size = (height, width)
    arrays = []

    for i in range(len(imgpaths)):
        img = image.load_img(imgpaths[i], target_size=size)
        arr = image.img_to_array(img)
        arrays.append(arr)

    return np.array(arrays)

