import cv2
import json 
import numpy as np 
from example import get_mask


def main():
    cho1 = cv2.imread('image processing/Exercises/Task 3/imgs/cho1.jpg')
    with open('image processing/Exercises/Task 3/json/cho1.json', 'r') as file:
        arr = json.loads(file.read())

    li = []
    w, h = arr['shape']['width'], arr['shape']['height']
    for i in arr['segmentation']:
        li.extend(np.arange(i[0], i[0] + i[1]))

    data = []
    for i in li:
        row = int(i // h)
        col = int(i - h * row)
        data.append([[row, col]])

    mask = get_mask(cho1, data)
    im = cv2.bitwise_and(cho1, cho1, mask=mask)

    cv2.imshow('mask',mask)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    

