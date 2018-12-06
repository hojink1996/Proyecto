from keras.applications import resnet50
from keras import models
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from Tools.retrain_adversarial import generate_new_model
from Tools.retrain_adversarial import expected_answers
from Tools.load_single_imagenet import n_arrays_adversarial

# Path to answers
path = '/home/hojin/Documentos/Primavera 2018/Inteligencia/Proyecto/val.txt'

# Variables para reentrenar
n_epochs = 1
batch_size = 1

# Load RESNET50
resnet_model = resnet50.ResNet50(weights = 'imagenet')

# Create new model for RESNET50
# generate_new_model(resnet_model, 'newresnet50.h5')

# Load model for RESNET50
new_model = models.load_model('newresnet50.h5')

# Expected values
y_expected = expected_answers(path, 2000)
y_train = y_expected
# y_val = y_expected[10:]

# Set the Gradient Descent
sgd = SGD(lr=5e-3, momentum=0.9, decay=1e-6, nesterov=True)
new_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Get the 2000 first adversarial examples
train_adv = n_arrays_adversarial(0, 2000, 224, 224)
# val_adv = n_arrays_adversarial(11, 20, 224, 224)

# Save best model with Early Stopping
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')

callbacks_list = [checkpoint, early_stop]

# Fit the model
train_adv = resnet50.preprocess_input(train_adv)
record = new_model.fit(x=train_adv, y=y_train, epochs=n_epochs, shuffle=True, verbose=1, callbacks=callbacks_list,
                       validation_split=0.1)
generate_new_model(new_model, '2000examples5epochs.h5')

# Get predictions
# pred = new_model.predict(val_adv)
# pred_original = resnet_model.predict(val_adv)

