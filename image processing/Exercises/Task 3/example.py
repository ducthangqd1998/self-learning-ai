import cv2
import json 
import numpy as np 


cho2 = cv2.imread('image processing/Exercises/Task 3/imgs/cho2.jpg')

def get_mask(img, contours):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.asarray(contours)], 255)

    return mask

with open('image processing/Exercises/Task 3/json/example.json', 'r') as file:
    data = json.loads(file.read())

mask = get_mask(cho2, data)
im = cv2.bitwise_and(cho2, cho2, mask=mask)


cv2.imshow('mask',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
