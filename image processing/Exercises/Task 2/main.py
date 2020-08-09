import json
import matplotlib.pyplot as plt 
import numpy as np
import cv2 


with open('image processing/Exercises/Task 2/task2.json', 'r') as file:
    arr = json.loads(file.read())
# ------------------- matplotlib ------------------
# plt.plot(arr[0], arr[1], 'o')
# plt.show()


# ------------------- opencv -----------------------------
img = np.zeros((700, 700))
contours = list()
for i in range(len(arr[1])):
    contours.append([[arr[0][i], 900 - arr[1][i]]])

cv2.drawContours(img, np.asarray(contours), -1, 255, 3)

cv2.imshow('Contours', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()