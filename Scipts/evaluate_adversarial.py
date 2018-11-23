"""
Script that evaluates the Accuracy of the Model over the Adversarial Examples.

Authors: Hojin Kang and Tomas Nunez
"""

from Tools.load_single_imagenet import n_images_validation, n_images_adversarial
import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image

from Tools.evaluate import eval_top5, eval_top1


# Save the top 5 predictions for each example
top5_pred_normal = []
top1_pred_normal = []

# Number of images
n_images = 450

# Load the model
resnet_model = resnet50.ResNet50(weights='imagenet')

# Get the original classes and the adversarial images
_ , clases, identifiers = n_images_validation(0, n_images, 224, 224)
images = n_images_adversarial(0, 450, 224, 224)

# Loop over the images
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

# Print the Accuracy
print('Accuracy Top 5 (Adversarial): ' + str(eval_top5(identifiers, top5_pred_normal)))
print('Accuracy Top 1 (Adversarial): ' + str(eval_top1(identifiers, top1_pred_normal)))