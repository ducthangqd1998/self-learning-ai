import cv2
import time
import random
import numpy as np
import tensorflow as tf


class ToTensor(object):

    def __call__(self, img, mask):
        """
        Convert Image to Tensorflow Tensor
        :param img:
        :param mask:
        :return: img, mask
        """
        img = tf.convert_to_tensor(img)
        mask = tf.convert_to_tensor(mask)
        return img, mask


class Normalize(object):
    def __init__(self, std, mean):
        super(Normalize, self).__init__()
        self.std = std
        self.mean = mean

    def __call__(self, img, mask):
        """
        Normalize image with mean and std of imagenet dataset
        :param img:
        :param mask
        :return: img, mask
        """
        img = img / 255.0
        mask = mask / 255.0
        img -= self.mean
        img /= self.std
        return img, mask


class RandomFlip(object):

    def __call__(self, img, mask):
        """
        Flip image and mask, the image will be flipped hôintal, vertical or both
        :param img:
        :param mask:
        :return: img, mask
        """
        if random.choice([0, 1]):
            print(1)
            axis = random.choice([-1, 0, 1])
            img = cv2.flip(img, axis)
            mask = cv2.flip(img, axis)

        return img, mask


class Resize(object):
    def __init__(self, dim):
        super(Resize, self).__init__()
        self.dim = dim

    def __call__(self, img, mask):
        """
        Resize image with shape(width, height)
        :param img:
        :param mask:
        :return: img, mask
        """
        img = cv2.resize(img, self.dim)
        mask = cv2.resize(mask, self.dim)
        return img, mask


class RandomRotate(object):
    def __call__(self, img, mask):
        """
        Rotate image with random degree
        :param img:
        :param mask:
        :return:
        """
        degree = random.uniform(0, 360)
        h, w = img.shape[:2]
        c = cv2.getRotationMatrix2D((w/2, h/2), degree, scale=1.0)

        img = cv2.warpAffine(img, c, img.shape[:2])
        mask = cv2.warpAffine(mask, c, img.shape[:2])

        return img, mask


class RandomCrop(object):
    def __init__(self, dim):
        super(RandomCrop, self).__init__()
        self.dim = dim

    def __call__(self, img, mask):
        """
        Crop image with fixed shape and random
        :param img:
        :param mask:
        :return: img, mask
        """
        h, w = img.shape[:2]
        y = random.randint(0, h - self.dim[1])
        x = random.randint(0, w - self.dim[0])

        img = img[y:y+self.dim[1], x:x + self.dim[0]]
        mask = mask[y:y + self.dim[1], x:x + self.dim[0]]

        return img, mask


class RandomGaussianBlur(object):

    def __call__(self, img, mask):
        '''
        Blur image using Gaussian with random radius
        :param img:
        :param mask:
        :return: img, mask
        '''
        if random.choice([0, 1]):
            radius = random.choice([1, 3, 5])
            img = cv2.GaussianBlur(img, (radius, radius), 0)

        return img, mask


class ChangeBrightness(object):

    def __call__(self, img, mask):
        """
        Change Brightness of image, image can be brighter, darker or not
        :param img:
        :param mask:
        :return: img, mask
        """
        if random.choice([0, 1]):
            lookUpTable = np.empty((1, 256), np.uint8)
            gamma = random.uniform(0.1, 3)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            img = cv2.LUT(img, lookUpTable)

        return img, mask


import json
from config import *
from os.path import join
def main():
    with open('dataset/data.json', 'r') as f:
        arr = json.loads(f.read())

    data = arr['test']
    for img_name in data:
        print(img_name)
        img_path = join(img_paths, 'Testing', img_name)
        mask_path = join(mask_paths, 'Testing', img_name.split('.')[0] + '.png')
        print(mask_path, img_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        img, mask = Resize(dim=(256, 256))(img, mask)
    # start_time = time.time()
    # img = cv2.imread('dataset/Original/Testing/Frame00010-org.jpg')
    # mask = cv2.imread('dataset/MASKS/Testing/Frame00010-org.png')
    #
    #
    # img, mask = Resize(dim=(256, 256))(img, mask)
    # img, mask = RandomRotate()(img, mask)
    # img, mask = ChangeBrightness()(img, mask)
    # img, mask = RandomGaussianBlur()(img, mask)
    # img, mask = RandomCrop(dim=(224, 224))(img, mask)
    # img, mask = Normalize(std=[0.229, 0.224, 0.225],
    #                       mean=[0.485, 0.456, 0.406])(img, mask)
    # # img, mask = ToTensor()(img, mask)
    # print("Execute time:", time.time() - start_time)
    #
    # # hair = cv2.bitwise_and(img, img, mask=mask[:, :, 0])
    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # # cv2.imshow('hair', hair)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()