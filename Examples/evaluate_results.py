from load_single_imagenet import n_images_validation
import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image

def eval_top5(pred_reales, top5_obtenido):
    count = 0.
    total = 0.
    i = 0
    for top5_temp in top5_obtenido:
        pred_real = pred_reales[i]
        if pred_real in top5_temp:
            count = count + 1
        total = total + 1
        i = i + 1
    return count/total*100

def eval_top1(pred_reales, top1_obtenido):
    count = 0.
    total = 0.
    i = 0
    for top1_temp in top1_obtenido:
        pred_real = pred_reales[i]
        if pred_real == top1_temp:
            count = count + 1
        total = total + 1
        i = i + 1
    return count/total*100

top5_pred_normal = []
top1_pred_normal = []
resnet_model = resnet50.ResNet50(weights='imagenet')

images, clases, identifiers = n_images_validation(0, 450, 224, 224)

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

print('Accuracy Top 5 (Original): ' + str(eval_top5(identifiers, top5_pred_normal)))
print('Accuracy Top 1 (Original): ' + str(eval_top1(identifiers, top1_pred_normal)))