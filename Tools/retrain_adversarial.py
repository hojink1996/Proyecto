"""
Tools used to retrain the model using adversarial examples.

Authors: Hojin Kang and Tomas Nunez
"""
from keras import models
import numpy as np


def generate_new_model(model, name):
    """
    Generate a new model to retrain an old model and saves it

    :param model:   The model to save
    :param name:    Name of the file to save
    """
    # Create a new Model
    new_model = models.Sequential()

    # Add the received model
    new_model.add(model)
    new_model.summary()

    # Save the new model
    new_model.save(name)


def generate_value_map(path_to_tags):
    """
    Generate a dictionary with the tags and its values

    :param path_to_tags:    Path to the txt containing the tags
    :return:                A dictionary representing the tags and its values
    """
    # Dictionary with the tags
    identificadores = {}
    with open(path_to_tags) as tag:
        content = tag.readlines()

    # Link every directory with it's word
    for palabra in content:
        linea = palabra.split('\t')
        codigo = linea[0]
        descriptor = linea[1].strip('\n')
        identificadores[codigo] = descriptor

    return identificadores


def expected_answers(path_to_val, num):
    """
    Generate the expected output as a One Hot vector

    :param path_to_val:     Path to the txt containing the expected values
    :param num:             Number of the value
    :return:                A One Hot Vector representing the expected output
    """
    # Number of classes
    n_classes = 1000

    # Expected values
    expect_vals = []

    # Get the content
    with open(path_to_val) as vals:
        content = vals.readlines()

    i = 0
    # Get the expected values
    for result in content:
        if(i >= num):
            break
        resultado = result.split(' ')[1]
        expect_vals.append(int(resultado[:-1]))
        i = i + 1

    # Get the one hot representation of the values
    one_hot_expect = np.eye(n_classes)[np.array(expect_vals)]

    return one_hot_expect
