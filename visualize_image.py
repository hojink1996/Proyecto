from load_images import image_set, image_word
import matplotlib.pyplot as plt

# Image to visualize
n = 35788

# Visualize an image
imagen =image_set[n]
imgplot = plt.imshow(imagen)
plt.show()
print(image_word[n])