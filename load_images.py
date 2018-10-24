import glob
import cv2
import matplotlib.pyplot as plt

# Location of file with tags and images
tags = glob.glob('/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/words.txt')
images = glob.glob('/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/tiny-imagenet-200/train/**/*.JPEG',
                   recursive = True)

# Read lines from the text file with the tags
with open(tags) as tag:
    content = tag.readlines()

# Save images
image_set = []
for img in images:
    image = cv2.imread(img)
    image_set.append(image)

# Visualize an image
imagen =image_set[100]
imgplot = plt.imshow(imagen)
plt.show()
