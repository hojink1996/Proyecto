# Proyecto EL-4106: Defensa Adversaria

#### Members: Hojin Kang y Tomás Nuñez

## Codes in the repository

#### Tools
<pre>
Scripts that have the tools required for the evaluation

adv_example_generation.py   --  Has all the scripts needed to generate adversarial
                                examples and visualize them, whether individually
                                or in batches. Also can save the adversarial examples.

evaluate.py                 --  Has the functions necessary to evaluate the Accuracy
                                of the models in a given dataset.
                                                                
load_single_imagenet.py     --  Has various scripts to load images from the ImageNet
                                dataset. Also has the processing of images.
                                
single_image.py             --  To return a single resized image and its tag.

</pre>

#### Scripts

<pre>
Scripts that have the implementation using the Tools

evaluate_adversarial.py     --  Evaluates the model with the adversarial examples.

evaluate_results.py         --  Evaluates the model with the original images.

preimplemented_imagenet.py  --  Sample for generating a single adversarial example.
                                Gets the original image and its prediction, and
                                compares it to the filtered image and its predictions.

test_imagenes.py            --  Generates the adversarial examples.
          
visualize_image.py          --  Visualize a single image (resized) and its tag 
                                (doesn't load all images)
</pre>
## Instructions

For simple generation and visualization of adversary examples use <i>preimplemented_imagenet.py</i>.
Adjust the value of <i>n</i> to get different images.

<b>NOTE:</b> Change the paths in lines 15 and 16 of <i>load_single_imagenet.py</i> to the
correct path of the file containing the words and the file containing the link to the images
respectively.
## Example case for image visualization

![Example image](Examples/example_image.png)

![Example text](Examples/example_text.png)

## Example case for adversarial example

#### Original image and prediction
![Example image](Examples/original_image.png)

![Example image](Examples/original_pred.png)

#### Filter for Adversarial Example
![Example image](Examples/filter.png)

#### Adversarial image and prediction
![Example image](Examples/adversarial_example.png)

![Example image](Examples/adversarial_pred.png)

## Results obtained with FGSM

#### Accuracy for original images

![Example image](Examples/original_accuracy.png)

#### Accuracy for adversarial examples

![Example image](Examples/adversarial_accuracy.png)

## Retraining the Model

#### Description of Models

<pre>
All models are fine tuned versions of ResNet50. The main difference
between the models is the way they were trained, which we will describe
in the following section.


<b>===== First Model =====</b>

Learning Rate = 5e-3
Momentum = 0.9
Decay = 1e-6
Dataset = 5000 Images
Validation Split = 0.1
100 Epochs con Early Stopping (5 Epochs)
Val Acc = 0.430


<b>===== Second Model =====</b>

Learning Rate = 5e-3
Momentum = 0.9
Decay = 1e-6
Dataset = 6500 Images
Validation Split = 0.2
100 Epochs con Early Stopping (5 Epochs)
Val Acc = 0.456

<b>===== Third Model =====</b>
Learning Rate = 1e-3
Momentum = 0.9
Decay = 1e-6
Dataset = 7500 Images
Validation Split = 0.2
100 Epochs con Early Stopping (5 Epochs)
Val Acc = 0.537

<b>===== Fourth Model =====</b>
Learning Rate = 5e-4
Momentum = 0.9
Decay = 1e-6
Dataset = 8500 Images
Validation Split = 0.2
100 Epochs con Early Stopping (5 Epochs)

<b>===== Fifth Model =====</b>
Learning Rate = 1e-3
Momentum = 0.9
Decay = 1e-6
Dataset = 8500 Images
Validation Split = 0.2
100 Epochs con Early Stopping (5 Epochs)


</pre>

With the previously described models,we obtain the following learning
curves:

![Example image](Examples/learning_curve_total.png)

#### Accuracy Inception v3 on Adversarial Dataset

Top 1 Accuracy: 55.6%   (Decrease from 78.8%)

Top 5 Accuracy: 81.3%   (Decrease from 94.4%)
