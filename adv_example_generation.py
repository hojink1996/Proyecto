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

def deepfool(x, model, eps=1e-6, max_iter=100, classes=1000):
    tol = 1e-8
    xadv = x.copy()
    # Predicted result in normal case
    y = model.predict(x)
    y_class = y.argmax()

    #Initialize current class as class predicted from raw input
    y_class_i = y_class

    #Set loss function as cross entropy
    gradientes = [K.gradients(K.softmax(model.output.op.inputs)[:, i], model.input)[0] for i in range(classes)]

    #Build function that computes gradient of loss function
    val_gradiente = K.function([model.input], gradientes)
    grad = val_gradiente([x])[0]

    #Initialize iteration counter and perturbation
    nb_iter = 0
    perturb = xadv
    while y_class_i == y_class and nb_iter < max_iter:
        grd_dif = grad - grad[y_class]
        y_diff = y - y[y_class]

        #Mask the true label (not considered when calculating perturbation norm)

        mask = [0]*classes
        mask[y_class] = 1
        norm = np.linalg.norm(grd_dif.reshape(classes,-1), axis=1)
        diff_normalized = np.ma.array(np.abs(y_diff)/norm, mask=mask)

        #Choose index of smallest difference, fill value corresponding to true class to +inf
        l = diff_normalized.argmin(fill_value =np.inf)
        r = (abs(y_diff[l]) / (pow(np.linalg.norm(y_diff[l]), 2) + tol)) * y_diff[l]
        perturb = np.clip(perturb + r, 0, 1)

        #Recalculate prediction for potential adversarial example

        y = model.predict(perturb)
        grad = val_gradiente([perturb])
        y_class_i = y.argmax()
        nb_iter += 1

    #Adversarial example as a function of eps
    xadv = np.clip(xadv + (1 + eps)*(perturb - xadv),0,1)

    #Label assigned to adversarial example
    y_adv = model.predict(xadv).argmax()

    #Return adversarial examples, and both initial and adversarial prediction

    return xadv, y_class, y_adv

def arraytoimage(xarr, dim):
    """
    Makes a PIL image from an array

    :param      xarr    : An array corresponding to the image
                dim     : The dimensions of the image

    :return:    The PIL image that represents the original image
    """
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