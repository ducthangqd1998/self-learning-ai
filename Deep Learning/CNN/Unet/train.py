import os
import cv2
import json
import argparse
import numpy as np
import tensorflow as tf
from os.path import join
from glob import glob
from config import *
from dataset import Dataset
from model import Unet
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def load_json(json_file):
    with open(json_file, 'r') as file:
        return json.loads(file.read())


def write_json(json_file, arr):
    with open(json_file, 'w') as file:
        json.dump(arr, file, indent=4)
##

def main(args):
    data = load_json(json_path)
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    test_loader = Dataset(data['test'], phase='Testing', batch_size=batch_size)
    train_loader = Dataset(data['train'], phase='Training', batch_size=batch_size)

    print(len(test_loader))

    model = Unet()

    losses = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    metric = tf.keras.metrics.MeanIoU(num_classes=1)

    # model.compile(optimizer=optimizer, loss=losses, metrics=metric)
    #
    # model.fit_generator(train_loader,
    #                     steps_per_epoch=len(train_loader),
    #                     epochs=epochs,
    #                     validation_data=test_loader,
    #                     validation_steps=len(test_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hair Segmentation Model - TF 2.3')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train')
    args = parser.parse_args()

    main(args)



