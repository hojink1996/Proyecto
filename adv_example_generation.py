"""
Script that generates adversarial examples and also does the reverse procedure of the preprocessing.

Authors: Hojin Kang and Tomas Nunez
"""

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import metrics
import numpy as np
from PIL import Image


def fast_gradient(model, x, eps=0.25):
    """
    Generates an adversarial example for the model using the fast gradient method

    :param      model   : The model from which to generate an adversarial example
    :param      x       : The original image from which to generate the adversarial example
    :param      eps     : The epsilon parameter that ponderates the gradient sign

    :return:    A tuple containing the adversarial example and the filter (sign of gradient) used to generate it
    """
    signo = np.zeros(np.shape(x))
    xadv = np.zeros(np.shape(x))
    for i, xi in enumerate(x):
        # Predicted result in normal case
        y = model.predict(xi).argmax()

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
        signoi = np.sign(val_gradiente([xi])[0])
        xadv[i] = xi + eps*signoi
        signo[i] = signoi

    return xadv, signo

# def fast_gradient_batch_generation(model, x, eps=0.25):
#     """
#     Generates an adversarial example for the model using the fast gradient method
#
#     :param      model   : The model from which to generate an adversarial example
#     :param      x       : An array of images from which to generate the adversarial examples
#     :param      eps     : The epsilon parameter that ponderates the gradient sign
#
#     :return:    A tuple containing the adversarial examples generated and the filters used to generate them
#     """
#     # Predicted result in normal case
#     y = model.predict(x).argmax()
#
#     # Make predicted variable into categorical variable (0s or 1s) of 1000 classes (ImageNet)
#     y_categorical = to_categorical(y, 1000)
#
#     # Make the predicted class the target
#     esperado = K.variable(y_categorical)
#
#     # Set the loss function as Cross Entropy (for Classification problem)
#     costo = metrics.categorical_crossentropy(model.output, esperado)
#
#     # Get the gradient of the function
#     gradiente = K.gradients(costo, model.input)
#     val_gradiente = K.function([model.input], gradiente)
#
#     xadv = []
#     filter = []
#     for x_image in x:
#         # Remember that the adversarial examples are x + eps*sign(gradient)
#         signo = np.sign(val_gradiente([x_image])[0])
#         xadv.append(x_image + eps*signo)
#         filter.append(signo)
#
#     return xadv, filter

def fast_gradient_batch_generation(model, x, eps=0.25):
    """
    Generates an adversarial example for the model using the fast gradient method

    :param      model   : The model from which to generate an adversarial example
    :param      x       : An array of images from which to generate the adversarial examples
    :param      eps     : The epsilon parameter that ponderates the gradient sign

    :return:    A tuple containing the adversarial examples generated and the filters used to generate them
    """
    xadv = []
    filter = []
    for x_image in x:
        x_image = np.asarray([x_image])
        # Predicted result in normal case
        y = model.predict(x_image).argmax()

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
        signo = np.sign(val_gradiente([x_image])[0])[0]
        xadv.append(x_image + eps*signo)
        filter.append(signo)

    return xadv, filter

def deepfool(x, model, eps=1e-6, max_iter=100, classes=1000, search_classes=31):
    """
    Generates an adversarial example for the model using DeepFool

    :param      x               : The original image from which to generate the adversarial example
    :param      model           : The model from which to generate an adversarial example
    :param      eps             : The epsilon parameter that ponderates the perturbation
    :param      max_iter        : Maximum number of iterations for DeepFool
    :param      classes         : Number of classes
    :param      search_classes  : Number of classes to search

    :return:    A tuple containing the adversarial example, the original class and the adversary class
    """
    tol = 1e-8
    xadv = x.copy()
    # Predicted result in normal case
    y = model.predict(x)[0]
    y_class = y.argmax()

    # Initialize current class as class predicted from raw input
    y_class_i = y_class

    # Build function that computes class gradient

    gradientes_search = [K.gradients(model.output[:, i], model.input)[0] for i in range(search_classes)]
    val_gradiente = K.function([model.input], gradientes_search)

    # Resize to compensate for classes not being in search_classes
    grad = np.swapaxes(np.array(val_gradiente([xadv])), 0, 1)[0]
    size = grad.shape
    sizePad = (1000,) + size[1:]
    gradPadded = np.zeros(sizePad)
    gradPadded[:grad.shape[0], :grad.shape[1], :grad.shape[2], :grad.shape[3]] = grad


    # grad = val_gradiente([x])[0]

    # Initialize iteration counter and perturbation
    nb_iter = 0
    perturb = xadv
    while y_class_i == y_class and nb_iter < max_iter:
        grd_dif = gradPadded - gradPadded[y_class]
        y_diff = y - y[y_class]

        # Mask the true label (not considered when calculating perturbation norm)

        mask = [0]*classes
        mask[y_class] = 1
        mask[search_classes:] = [1]*(classes-search_classes)
        norm = np.linalg.norm(grd_dif.reshape(classes,-1), axis=1) + tol
        diff_normalized = np.ma.array(np.abs(y_diff)/norm, mask=mask)

        # Choose index of smallest difference, fill value corresponding to true class to +inf
        l = diff_normalized.argmin(fill_value =np.inf)
        r = (abs(y_diff[l]) / (pow(np.linalg.norm(y_diff[l]), 2) + tol)) * y_diff[l]
        perturb = np.clip(perturb + r, 0, 1)

        # Recalculate prediction for potential adversarial example

        # Resize t
        y = model.predict(perturb)[0]
        grad = np.swapaxes(np.array(val_gradiente([xadv])), 0, 1)[0]
        gradPadded[:grad.shape[0], :grad.shape[1], :grad.shape[2], :grad.shape[3]] = grad
        y_class_i = y.argmax()
        nb_iter += 1

    # Adversarial example as a function of eps
    xadv = np.clip(xadv + (1 + eps)*(perturb - xadv),0,1)

    # Label assigned to adversarial example
    y_adv = model.predict(xadv).argmax()

    # Return adversarial examples, and both initial and adversarial prediction

    return xadv, y_class, y_adv


def arraytoimage(xarr, dim):
    """gradient sign
    Makes a PIL image from an array (removing preprocessing for the RESNET network)

    :param      xarr    : An array corresponding to the image
    :param      dim     : The dimensions of the image

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