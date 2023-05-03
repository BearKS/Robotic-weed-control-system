from extraction_main import center_box_position, crop_plant


if __name__ == "__main__":
    pos,crop_plant = bounding('demo4_1.png','demo4.png') # Binary Mask, Original image
    print(type(crop_plant))
    print(crop_plant)