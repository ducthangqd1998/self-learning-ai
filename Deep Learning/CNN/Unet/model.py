import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, UpSampling2D

def conv_block(filters, x):
    x = Conv2D(filters=filters, 
                kernel_size=(3,3), 
                padding='valid', 
                activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=filters, 
                kernel_size=(3,3), 
                padding='valid', 
                activation='relu')(x)
    out = BatchNormalization()(x)
    return out 


def down_sample(filters, x):
    x = MaxPool2D((2,2))(x)
    out = conv_block(filters, x)
    return out


def up_sample(filters, x, x1):
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()(x1, x)
    out = conv_block(filters, x)

class Unet(Model):
    def __init__(self):
        super(Unet, self).__init__()
        pool = MaxPool2D((2,2))
    def call(self, x):
        # Encoder 
        print(x)
        down1 = conv_block(64, x)
        down2 = down_sample(128, down1)
        down3 = down_sample(256, down2)
        down4 = down_sample(512, down3)
        down5 = down_sample(1024, down4)

        # Decoder
        up1 = up_sample(512, down5, down4)
        up2 = up_sample(256, up1, down3)
        up3 = up_sample(128, up2, down2)
        up4 = up_sample(64, up3, down1)

        out = Conv2D(filters=1, 
                    kernel_size=(1,1),
                    padding='valid',
                    activation='sigmoid')

        return out

image = np.zeros((1, 572, 572, 3))
img_tensor = tf.image.convert_image_dtype(image, dtype=tf.float16)

model = Unet()
model(img_tensor)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


        


