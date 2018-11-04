'''
Note: Does nothing at the moment.
TODO: Run a pretrained model.

Authors: Hojin Kang and Tomas Nunez
'''

import numpy as np
import glob
import os
from keras.applications import resnet50
from keras.preprocessing import image
from keras.applications import inception_v3
import matplotlib.pyplot as plt
import ssl

# Arreglar error de ssl
ssl._create_default_https_context = ssl._create_unverified_context


# 2 Models used: ResNet and Inception V3
resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
images = glob.glob('/home/tomas/Documents/Inteligencia Computacional/tiny-imagenet-200/train/**/*.JPEG',
                   recursive = True)
labelpath = '/home/tomas/Documents/Inteligencia Computacional/tiny-imagenet-200/words.txt'
n = 5015
with open(labelpath) as tag:
    content = tag.readlines()
# We make a HashMap (dictionary) with the identifiers
identificadores = {}

# Link every directory with it's word
for palabra in content:
    linea = palabra.split('\t')
    codigo = linea[0]
    descriptor = linea[1].strip('\n')
    identificadores[codigo] = descriptor

# Select n-th image, preprocess it for each model
imgpath = images[n]

# Input size for ResNet = 224*224
img = image.load_img(imgpath, target_size=(224, 224))

# Input size for Inception V3 = 299*299
imgInception = image.load_img(imgpath, target_size=(299, 299))

# Preprocessing
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = resnet50.preprocess_input(x)
xInc = image.img_to_array(imgInception)
xInc = np.expand_dims(xInc, axis=0)
xInc = inception_v3.preprocess_input(xInc)

# Get predictions from both models
pred = resnet_model.predict(x)
predInc = inception_model.predict(xInc)
corresponding_link = os.path.basename(imgpath)
corresponding_link = corresponding_link.split('_')[0]
clase = identificadores[corresponding_link]
print('Predicted ResNet:', resnet50.decode_predictions(pred, top=5)[0])
print('Predicted InceptionV3: ', inception_v3.decode_predictions(predInc, top=5)[0])
print('Real: ', clase)

implot = plt.imshow(img)
plt.show()