import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

names = ['input_image', 'conv1_relu', 'conv2_block1_1_relu', 'conv2_block3_2_relu', 'conv3_block4_2_relu']

# Xác định các layer cần lấy
# inputs = Input(shape=(224, 224, 3), name="input_image")
# backbone = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
# backbone.summary()

'''
Chúng ta sẽ lấy các layer theo các shape sau:
    layer 1: 64, h/2, w/2           => conv1_relu
    layer 2: 256, h/4, w/4          => conv2_block1_out
    layer 3: 512, h/8, w/8          => conv3_block2_out
    layer 4: 1024, h/16, w/16       => conv4_block3_out
    out layer: 2048, h/32, w/32     => conv5_block2_out
'''

class Unet(Model):
    def __init__(self):
        super(Unet, self).__init__()
        self.backbone = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        # self.input_layer = self.backbone.get_layer(name='input_image')
        self.layer0 = self.backbone.get_layer(name='conv1_relu')
        self.layer1 = self.backbone.get_layer(name='conv2_block1_out')
        self.layer2 = self.backbone.get_layer(name='conv3_block2_out')
        self.layer3 = self.backbone.get_layer(name='conv4_block3_out')
        self.out_layer = self.backbone.get_layer(name='conv5_block2_out')

    def call(self, input):
        encoder_out = self.out_layer.output
        x = self.up_sample(self.layer3, encoder_out, 1024)
        x = self.up_sample(self.layer2, x, 512)
        x = self.up_sample(self.layer1, x, 256)
        x = self.up_sample(self.layer0, x, 64)
        # x = self.up_sample(self.input_layer, x, 64)

        x = Conv2D(1, (1, 1), padding="same")(x)
        out = Activation("sigmoid")(x)
        return out

    def conv_block(self, filters, x):
        x = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu')(x)

        x = BatchNormalization()(x)

        x = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu')(x)
        out = BatchNormalization()(x)
        return out

    def up_sample(self, layer, input, filters):
        x1 = layer.output
        x = UpSampling2D((2, 2))(input)
        x = Concatenate()([x, x1])

        out = self.conv_block(filters, x)

        return out

    def summary(self):
        x = self.backbone.input
        return Model(inputs=x, outputs=self.call(x))

model = Unet()
model_func = model.summary()
model_func.summary()









