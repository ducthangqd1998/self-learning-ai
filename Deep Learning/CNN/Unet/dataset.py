import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from os.path import join
from config import *
from transforms import *
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, imgs, phase='Training', batch_size=8):
        self.phase = phase
        self.batch_size = batch_size
        self.imgs = imgs


    def load_data(self, imgs_batch):
        imgs = list()
        masks = list()

        for img_name in imgs_batch:
            img_path = join(img_paths, self.phase, img_name)
            mask_path = join(mask_paths, self.phase, img_name.split('.')[0] + '.png')

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            if self.phase == 'Training':
                img, mask = self.train_transform(img, mask)
            else:
                img, mask = self.val_transfrom(img, mask)

            imgs.append(img)
            masks.append(mask)

        imgs = tf.convert_to_tensor(imgs)
        masks = tf.convert_to_tensor(masks)

        return imgs, masks


    def train_transform(self, img, mask):
        img, mask = Resize(dim=(256, 256))(img, mask)
        img, mask = RandomRotate()(img, mask)
        img, mask = ChangeBrightness()(img, mask)
        img, mask = RandomGaussianBlur()(img, mask)
        img, mask = RandomCrop(dim=(224, 224))(img, mask)
        img, mask = Normalize(std=[0.229, 0.224, 0.225],
                              mean=[0.485, 0.456, 0.406])(img, mask)
        img, mask = ToTensor()(img, mask[:, :, 0])

        return img, mask


    def val_transfrom(self, img, mask):
        img, mask = Resize(dim=(256, 256))(img, mask)
        img, mask = RandomCrop(dim=(224, 224))(img, mask)
        img, mask = Normalize(std=[0.229, 0.224, 0.225],
                              mean=[0.485, 0.456, 0.406])(img, mask)
        img, mask = ToTensor()(img, mask[:, :, 0])

        return img, mask


    def __getitem__(self, index):

        stop = (index + 1)*self.batch_size

        if stop > len(self.imgs):
            stop = len(self.imgs)

        imgs_batch = self.imgs[index*self.batch_size:stop]
        imgs, masks = self.load_data(imgs_batch)

        return imgs, masks


    def __len__(self):
        return int(np.ceil(len(self.imgs) / self.batch_size))


def main():
    img_names = list()
    img_paths = glob(join('dataset/Original/Testing', '*'))

    for i in img_paths:
        name = i.split('/')[-1]
        img_names.append(name)

    val_dataset = Dataset(img_names, phase='Testing', batch_size=8)

    for i in range(len(val_dataset)):
        img, mask = val_dataset[i]
        print(i, len((img)), len(img_names))


if __name__ == '__main__':
    main()


