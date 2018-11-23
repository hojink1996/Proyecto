"""
Script that visualizes a single image (given by n), and prints its tag.

Authors: Hojin Kang and Tomas Nunez
"""

from Tools.load_single_imagenet import single_img
import matplotlib.pyplot as plt

# Image to visualize
n = 132512

# Visualize an image
imagen, image_word = single_img(n, 224, 224)
imgplot = plt.imshow(imagen)
plt.show()

# Show the tag
print(image_word)