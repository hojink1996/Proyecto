from adv_example_generation import fast_gradient_batch_saving_no_return
import matplotlib.pyplot as plt
from keras.applications import resnet50
import numpy as np
from load_single_imagenet import single_img_val

resnet_model = resnet50.ResNet50(weights='imagenet')
for i in range(10):
    fast_gradient_batch_saving_no_return(resnet_model, 5, 4, True, 400 + i*20)
#
# for adv in adversarios:
#     img = plt.imshow(adv)
#     plt.show()
#     pred = resnet_model.predict(adv.reshape((1, 224, 224, 3)))
#     print('Predicted ResNet:', resnet50.decode_predictions(pred, top=5)[0])
# print('============= REAL =============')
# for clase in clases:
#     print(clase)