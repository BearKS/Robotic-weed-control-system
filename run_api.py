import requests
import cv2
img = cv2.imread('demo4.png')
print(img.shape)
response = requests.post('http://35.219.101.169:5000/', files = {'image': open('demo4.png', 'rb')})
print(response.json())