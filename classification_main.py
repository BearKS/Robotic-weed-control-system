from extraction_main import center_box_position, crop_plant
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Load the model
model = keras.models.load_model('model_classification.h5')

# Prediction function
def predict_weed_positions(crop_plant_list, center_box_position_list, model, threshold=0.5):
    weed_positions = []

    for idx, image in enumerate(crop_plant_list):
        # Ensure the image is the correct size (224x224)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)

        # Convert the image dtype to float32 before dividing
        image = image.astype(np.float32)
        image /= 255.0

        # Make predictions
        prediction = model.predict(image)

        # Determine if it's a weed or a crop
        if prediction > threshold:
            print("Image at index", idx, "Sweet Basil")
        else:
            print("Image at index", idx, "Weed")
            weed_positions.append(center_box_position_list[idx])

    return weed_positions

# Replace this with the list of input image arrays and their corresponding center box positions
weed_positions = predict_weed_positions(crop_plant, center_box_position, model)
print("The weed positions are:", weed_positions)