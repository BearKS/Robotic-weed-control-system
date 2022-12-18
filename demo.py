import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image

import os
os.environ["SM_FRAMEWORK"] = "tf.keras" #before the import
# %env SM_FRAMEWORK=tf.keras
# %pip install albumentations>=0.3.0 
# %pip install --pre segmentation-models

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig('result.png')
    plt.show()

    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

import segmentation_models as sm


BACKBONE = 'vgg16'
BATCH_SIZE = 4
CLASSES = ['plant']
#คลาสคืออะไร
LR = 0.0001
EPOCHS = 60

preprocess_input = sm.get_preprocessing(BACKBONE)
# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

from keras.models import load_model
model.load_weights('best_model.h5')
# model.summary()

aug = A.PadIfNeeded(min_height=1088, min_width=1920, p=1)

image = cv2.imread('RWSIP-1-1-65-1939.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

augmented = aug(image=image)
image_padded = augmented['image']

print(f'input : {image.shape}')

image_padded = np.expand_dims(image_padded, axis=0)
result = model.predict(image_padded).round()
# extraction = image_padded
extraction = cv2.imread('Extraction_result.png')
extraction = cv2.cvtColor(extraction, cv2.COLOR_BGR2RGB)


visualize(
    Input=image_padded.squeeze(),
    Segmentation_result = result[..., 0].squeeze(),
    Extraction_result = extraction.squeeze(),
    )

img =  cv2.imread('result.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img.squeeze())

img.show()

print('-----------SUCCESS----------')