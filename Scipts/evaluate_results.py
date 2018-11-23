"""
Script that evaluates the Accuracy of the Model over the original images.

Authors: Hojin Kang and Tomas Nunez
"""

from Tools.load_single_imagenet import n_images_validation
import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image

from Tools.evaluate import eval_top5, eval_top1

# Number of images
n_images = 450

# Save the Top 5 and Top 1 predictions
top5_pred_normal = []
top1_pred_normal = []

# Load the model
resnet_model = resnet50.ResNet50(weights='imagenet')

# Get the images, classes and identifieres
images, clases, identifiers = n_images_validation(0, n_images, 224, 224)

# Get the predictions for each image
for img in images:
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)
    pred = resnet_model.predict(x)
    top5_temp = []
    for i in range(5):
        if i == 0:
            top1_pred_normal.append(resnet50.decode_predictions(pred, top=5)[0][i][0])
        top5_temp.append(resnet50.decode_predictions(pred, top=5)[0][i][0])
    top5_pred_normal.append(top5_temp)

# Print Accuracy
print('Accuracy Top 5 (Original): ' + str(eval_top5(identifiers, top5_pred_normal)))
print('Accuracy Top 1 (Original): ' + str(eval_top1(identifiers, top1_pred_normal)))