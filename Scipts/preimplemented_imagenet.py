"""
Takes a image from Image Net and compares the classification result of the original image
and an adversarial example generated for the ResNet50 and Inception v3 models.
Also shows the original image, the filter used to generate the adversarial example
and the adversarial example itself.

Authors: Hojin Kang and Tomas Nunez
"""

# Model import
from keras.applications import resnet50
from keras.preprocessing import image
from keras.applications import inception_v3

# Extra library imports
import matplotlib.pyplot as plt
import ssl
import numpy as np

# Personal library import
from Tools.adv_example_generation import arraytoimage, fast_gradient_batch_generation
from Tools.load_single_imagenet import single_img_val


# Fix SSL Error
ssl._create_default_https_context = ssl._create_unverified_context


# 2 Models used: ResNet and Inception V3
resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
#n = 790245
# n = []
# for i in range(10):
#     n.append(random.randint(1,800000))
# Input size for ResNet = 224*224
inputs = np.zeros((4, 224, 224, 3))
labels = []
identifiers = []
images = []
for i in range(4):
    img, tag, identifier = single_img_val(i+1, 224, 224)

    # Input size for Inception V3 = 299*299
    # imgInception, tag, identifier = single_img(i, 299, 299)

    # Preprocessing
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)
    labels.append(tag)
    images.append(img)
    identifiers.append(identifier)
    inputs[i] = x



# xInc = image.img_to_array(imgInception)
# xInc = np.expand_dims(xInc, axis=0)
# xInc = inception_v3.preprocess_input(xInc)

# Get predictions from both models
pred = resnet_model.predict(inputs)
# predInc = inception_model.predict(xInc)
for j, clase in enumerate(labels):
    print('Predicted ResNet:', resnet50.decode_predictions(pred, top=5)[j])
    # print('Predicted InceptionV3: ', inception_v3.decode_predictions(predInc, top=5)[0])
    print('Real: ', clase, ' ', identifiers[j])

    # Show original imageimg = Image.fromarray(data, 'RGB')
    implot = plt.imshow(images[j])
    plt.show()


# Adversarial examples
print('===== Adversarial Examples ======')

# Generate an adversarial example for the resnet model

xadv, filter = fast_gradient_batch_generation(resnet_model, inputs, 3)
pred_adv = []
for i, xadv_i in enumerate(xadv):
    pred = resnet_model.predict(xadv_i)
    pred_adv.append(pred)
    filtplot = plt.imshow(filter[i])
    plt.show()
    adversarial_image = arraytoimage(xadv_i, (224, 224, 3))
    adversarialplot = plt.imshow(adversarial_image)
    plt.show()
    print('Predicted ResNet:', resnet50.decode_predictions(pred, top=5)[0])

# Testing
# xadv, _, pred = deepfool(x, resnet_model, classes=1000, search_classes=30)

# Show adversarial example filter


# Show adversarial example


# Show perturbation
# perturb = xadv-x
# filtplot = plt.imshow(perturb)
# plt.show()

