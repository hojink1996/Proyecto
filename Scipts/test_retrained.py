from keras import models
from keras.applications import resnet50

from Tools.retrain_adversarial import generate_new_model
from Tools.retrain_adversarial import expected_answers
from Tools.load_single_imagenet import n_arrays_adversarial

# Path to answers
path = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/val.txt'

retrained_model = models.load_model('bestweights.hdf5')

# Get the 2000 first adversarial examples
train_adv = n_arrays_adversarial(0, 100, 224, 224)

# Expected values
y_expected = expected_answers(path, 100)

# Get predictions
pred = retrained_model.predict(train_adv)

for i in range(100):
    print('Predicted ResNet:', resnet50.decode_predictions(pred, top=5)[i])
    print('Real Value:', resnet50.decode_predictions(y_expected, top=5)[i])