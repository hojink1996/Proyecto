import numpy as np
import glob
import os
import tensorflow as tf
from keras.applications import resnet50
from keras.preprocessing import image
from keras.applications import inception_v3
import matplotlib.pyplot as plt
import ssl

resnet_model = resnet50.ResNet50(weights='imagenet')
resnet_model.compile(loss='mean_squared_error', optimizer='sgd')
resnet_model.optimizer.get_gradients(resnet_model.total_loss, resnet_model.trainable_weights)
ybar = resnet_model.predict()

def fast_gradient(model, x, eps=0.01):
    xadv = tf.identity(x)
    ybar = model.predict(xadv)
    yshape = ybar.get_shape().as_list()
    yclasses = yshape[1]
    indices = tf.argmax(ybar, axis=1)
    targetLabels = tf.one_hot(indices, yclasses, on_value=1.0, off_value=0.0)
    loss_function = tf.nn.softmax_cross_entropy_with_logits
    eps = tf.abs(eps)
    ybar, logits = model(xadv, logits=True)
    loss = loss_function(labels=targetLabels, logits=logits)
    grad = tf.gradients(loss, xadv)
    xadv = tf.stop_gradient(xadv + eps*tf.sign(grad))
    xadv = tf.clip_by_value(xadv, 0.0, 1.0)
    return xadv