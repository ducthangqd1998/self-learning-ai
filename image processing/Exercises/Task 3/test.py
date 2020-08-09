import cv2
import numpy as np
import json 


cho1 = cv2.imread('image processing/Exercises/Task 3/cho1.jpg')
cho2 = cv2.imread('image processing/Exercises/Task 3/cho2.jpg')
cho1_mask = cv2.imread('image processing/Exercises/Task 3/cho1.png', -1)
cho2_mask = cv2.imread('image processing/Exercises/Task 3/cho2.png', -1)


mask1 = cho1_mask[:, :, 3]
mask2 = cho2_mask[:, :, 3]

_, contours, hierarchy = cv2.findContours(mask2.astype('uint8'),
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

h, w = mask2.shape
print(h, w)
arr = contours[0]

for i in arr:
    print(i)

arr1 = []
pre = 0
count = 0
check = False
for arr in contours:
    for index, value in enumerate(arr):
        x, y = value[0]
        num = x*h + y
        if num == pre + 1:
            try:
                if pre not in arr1[count]:
                    arr1[count].append(pre)
                arr1[count].append(num)
                check = True
            except:
                arr1.append([])
                if pre not in arr1[count]:
                    arr1[count].append(pre)
                arr1[count].append(num)
                check = True
        else:
            if check:
                count += 1
                check = False
            else:
                try:
                    arr1[count].append(pre)
                except:
                    arr1.append([])
                    arr1[count].append(pre)
                count += 1
        pre = num

data = {}
data['shape'] = {
    'width': w,
    'height': h
}

li = []

for i in arr1:
    li.append([int(i[0]), len(i)])

data['segmentation'] = li

with open('image processing/Exercises/Task 3/cho2.json', 'w') as file:
    json.dump(data, file, sort_keys=True, indent=4)

    



# # img1 = np.zeros(cho1.shape[:2], np.uint8) + 255

# cv2.drawContours(img1, contours, -1, (0, 255, 0), 1) 
  
# cv2.imshow('Contours', img1.astype('uint8')) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

# print(len(contours[0]))

