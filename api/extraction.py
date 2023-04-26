import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt

def denoise_image(img):
    # Load image in grayscale
    # img = cv2.cvtColor(img,cv2.)
    # Check if image was loaded successfully
    if img is None:
        # print('Error: Failed to load image at path:', )
        return None
    else:
        # Define structuring elements
        img = img.astype(np.uint8)
        img = cv2.bitwise_not(img)
        kernel_rect = np.ones((10, 10), np.uint8)
        kernel_circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        # Apply morphology operations with different kernels

        # dilated_rect = cv2.dilate(img, kernel_rect, iterations=1)
        eroded_rect = cv2.erode(img, kernel_rect, iterations=2)
        dilated_rect = cv2.dilate(eroded_rect, kernel_rect, iterations=2)
        # dilated_circ = cv2.dilate(img, kernel_circ, iterations=1)
        eroded_circ = cv2.erode(img, kernel_circ, iterations=2)
        dilated_circ = cv2.dilate(eroded_circ, kernel_circ, iterations=2)
        # dilated_cross = cv2.dilate(img, kernel_cross, iterations=1)
        eroded_cross = cv2.erode(img, kernel_cross, iterations=2)
        dilated_cross = cv2.dilate(eroded_cross, kernel_cross, iterations=2)
        # Combine the eroded images
        eroded = cv2.bitwise_or(eroded_rect, eroded_circ)
        eroded = cv2.bitwise_or(eroded, eroded_cross)
        # Combine the denoised images
        denoised = cv2.bitwise_or(dilated_rect, dilated_circ)
        denoised = cv2.bitwise_or(denoised, dilated_cross)
        print('denoised complete')
        cv2.imwrite('denoised.png',denoised)
        return denoised
    
def bounding(img, input):
    denoised_img = denoise_image(img)
    list_img = []
    if denoised_img is None:
        print('Error: Failed to load image at path:', img)
    else:
        gray_img = denoised_img
        # gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.astype(np.uint8)
        # gray_img = cv2.bitwise_not(gray_img)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

        # Find the contours in the processed image
        contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define the minimum area threshold
        max_area = 0

        # Define list for store position
        all_box_position = []
        center_box_position=[]
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            area = w*h
            max_area = max(max_area, area)

        # Set the minimum area threshold as the largest bounding box area
        min_area = max_area/70

        # # Draw the contours on the original image
        # int_img = cv2.imread(input)
        # int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2RGB)
        # mask = img/255
        # mask = mask.astype(np.uint8)
        # mask = cv2.bitwise_not(mask)
        result = input.copy()
        # result = cv2.bitwise_and(result,result,mask=mask)


        output_dir = 'crop_img'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        for contour in contours:
        # Calculate the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = w*h
            # Only draw the contour and bounding box if the area is larger than the minimum area threshold
            if area >= min_area:
                # Draw the bounding box
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add box position to the list
                all_box_position.append((x, y, w, h))
                center_box_position.append((x+w/2, y+h/2))
                # print(all_box_position,center_box_position)
                # Print position
                position_text = f"({x}, {y})"
                cv2.putText(result, position_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
                # Crop the image using the bounding box coordinates and save it as an RGB image
                cropped_image = result[y:y+h, x:x+w]
                list_img.append(cropped_image)
                # output_path = os.path.join(output_dir, f'cropped_{x}_{y}.png')
                cv2.imwrite(f'{contour}.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                # cv2.drawContours(result, contour, idx, color, 2)
        cv2.imwrite('output.png' , result)
        
        return center_box_position,list_img
 # Binary Mask, Original image