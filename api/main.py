from typing import Union
from fastapi import FastAPI,File, UploadFile
import numpy as np
import cv2
import segmentation
import extraction
import classification
import upload_file

app = FastAPI()

segmentation.load_model()
# classification.load_model()

def decodeByte2Numpy(inputImage):
    np_image = np.frombuffer(inputImage, np.uint8)
    np_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    # outputImage = np.array(inputImage)
    return np_image

def upload(img,file_name):
    bucket_name = "project-kmitl-image"
    destination_blob_name = f"image/{file_name}"
    
    upload_file.upload_image(bucket_name, img, destination_blob_name)
@app.post("/")
async def allProcess(image: UploadFile):
    print(image.filename)
    file_name = image.filename
    inputImage = await image.read()
    upload(inputImage,file_name)
    inputImage = decodeByte2Numpy(inputImage)
    segment = segmentation.inference(inputImage)
    pos,list_img,weed_pos = extraction.bounding(segment,inputImage)
    # weed_pos = classification.predict_weed_positions(list_img,pos)
    # print(inputImage.shape)
    return weed_pos

