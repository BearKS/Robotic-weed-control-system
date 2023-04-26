from typing import Union
from fastapi import FastAPI,File
import numpy as np
import cv2
import segmentation
import extraction
import classification

app = FastAPI()

segmentation.load_model()
# classification.load_model()

def decodeByte2Numpy(inputImage):
    outputImage = np.frombuffer(inputImage, np.uint8)
    outputImage = cv2.imdecode(outputImage, cv2.IMREAD_COLOR)
    return outputImage

@app.post("/")
def allProcess(image: bytes = File(...)):
    inputImage = decodeByte2Numpy(image)
    segment = segmentation.inference(inputImage)
    print(segment.shape)
    pos,list_img = extraction.bounding(segment,inputImage)
    # weed_pos = classification.predict_weed_positions(list_img,pos)
    # print(inputImage.shape)
    return pos


