'''
Script that generates adversarial examples and also does the reverse procedure of the preprocessing.

Authors: Hojin Kang and Tomas Nunez
'''

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import metrics
import numpy as np
from PIL import Image

def fast_gradient(model, x, eps=0.25):
    '''
    Generates an adversarial example for the model

    :param      model   : The model from which to generate an adversarial example
                x       : The original image from which to generate the adversarial example
                eps     : The epsilon parameter that ponderates the gradient sign

    :return:    A tuple containing the adversarial example and the filter (sign of gradient) used to generate it
    '''
    # Predicted result in normal case
    y = model.predict(x).argmax()

    # Make predicted variable into categorical variable (0s or 1s) of 1000 classes (ImageNet)
    y_categorical = to_categorical(y, 1000)

    # Make the predicted class the target
    esperado = K.variable(y_categorical)

    # Set the loss function as Cross Entropy (for Classification problem)
    costo = metrics.categorical_crossentropy(model.output, esperado)

    # Get the gradient of the function
    gradiente = K.gradients(costo, model.input)
    val_gradiente = K.function([model.input], gradiente)

    # Remember that the adversarial examples are x + eps*sign(gradient)
    signo = np.sign(val_gradiente([x])[0])
    xadv = x + eps*signo

    return xadv, signo

def arraytoimage(xarr, dim):
    '''
    Makes a PIL image from an array

    :param      xarr    : An array corresponding to the image
                dim     : The dimensions of the image

    :return:    The PIL image that represents the original image
    '''
    # Reshape the array to image dimensions
    x_out = np.reshape(xarr, dim)

    # Cancel the preprocessing
    x_out[..., 0] += 103.939
    x_out[..., 1] += 116.779
    x_out[..., 2] += 123.68

    # Convert from BGR to RGB (revert preprocessing)
    x_out = x_out[..., ::-1]

    # Convert array to image
    x_out = np.clip(x_out, 0., 255.).astype(np.uint8)
    img = Image.fromarray(x_out, 'RGB')

    return img