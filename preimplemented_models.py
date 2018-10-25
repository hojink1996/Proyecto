'''
TODO: Run a pretrained model.

Authors: Hojin Kang and Tomas Nunez
'''

import keras
import numpy as np
from keras.applications import resnet50

resnet_model = resnet50.ResNet50(weights='imagenet')