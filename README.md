# Proyecto EL-4106: Defensa Adversaria

#### Members: Hojin Kang y Tomás Nuñez

## Codes in the repository
<pre>
adv_example_generation.py   --  Has all the scripts needed to generate adversarial
                                examples and visualize them.
                                
load_single_imagenet.py     --  Loads a single image from the ImageNet library and
                                gives its tag and identifier. Image is loaded in
                                PIL format.
                                
preimplemented_imagenet.py  --  Sample for generating a single adversarial example.
                                Gets the original image and its prediction, and
                                compares it to the filtered image and its predictiong.
                                
single_image.py             --  To return a single resized image and its tag

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