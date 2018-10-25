'''
Script that visualizes a single image (given by n), and prints its tag.

Authors: Hojin Kang and Tomas Nunez
'''

from single_image import single_img
import matplotlib.pyplot as plt

# Image to visualize
n = 35788

# Visualize an image
imagen, image_word = single_img(n, 64, 64)
imgplot = plt.imshow(imagen)
plt.show()

# Show the tag
print(image_word)