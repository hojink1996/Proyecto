"""
Simple script that generates adversarial examples and saves them

Authors: Hojin Kang and Tomas Nunez
"""
from Tools.adv_example_generation import fast_gradient_batch_saving_no_return
from keras.applications import resnet50

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