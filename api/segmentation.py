import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
model = None
def load_model():
    global model
    n_classes = 1 
    # activation = 'sigmoid' if n_classes == 1 else 'softmax'
    def loss_fn():
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        return total_loss
    def IOUScore():
        return sm.metrics.IOUScore(threshold=0.5)
    def f1_fn():
        return sm.metrics.FScore(threshold=0.5)
    model = keras.models.load_model('../model/model_v10_128.h5',custom_objects={'dice_loss_plus_1binary_focal_loss':loss_fn,'iou_score':IOUScore,'f1-score':f1_fn})
    # return model

def inference(image):
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dim = (640,480)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    resized = np.expand_dims(resized, axis=0)
    result = model.predict(resized).round()
    # closing = cv2.morphologyEx(result[0], cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('mask.png' , result[0]*255 )
    # cv2.imwrite('mask1.png' , cv2.bitwise_not(np.round(result[0]))*255 )
    # result =  result.astype(np.uint32)
    # print(((result[0]*-1)+1)*255)
    # cv2.imwrite('mask1.png' ,((result[0]*-1)+1)*255)
    
    return result[0]*255
