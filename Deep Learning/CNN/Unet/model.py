import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, UpSampling2D


def conv_block(x, filters):
    x = Conv2D(filters=filters, 
                kernel_size=(3,3), 
                padding=(1,1), 
                activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters, 
                kernel_size=(3,3), 
                padding=(1,1), 
                activation='relu')(x)
    x = BatchNormalization()(x)
    return out 


def down_sample(x, filters):
    x = MaxPool2D((2,2))(x)
    out = conv_block(x)
    return out


def up_sample(x, x1, filters):
    x = UpSampling2D((2, 2))(x)
    x = Concatenate(x1, x)
    out = conv_block(x, filters)

class Unet(Model):
    def __init__(self):
        super(Unet, self).__init__()
        pool = MaxPool2D((2,2))
    def call(self, x):
        # Encoder 
        down1 = conv_block(64, x)
        down2 = down_sample(128, down1)
        down3 = down_sample(256, down2)
        down4 = down_sample(512, down3)
        down5 = conv_block(1024, down4)

        # Decoder
        up1 = up_sample(down5, down4, 512)
        up2 = up_sample(up1, down3, 256)
        up3 = up_sample(up2, down2, 128)
        up4 = up_sample(up3, down1, 64)

        out = Conv2D(filters=1, 
                    kernel_size=(1,1),
                    padding='same',
                    activation='sigmoid')

        return out

model = Unet()
image = np.zeros((1, 224, 224, 3))
img_tensor = tf.image.convert_image_dtype(image, dtype=tf.float16)
# model(img_tensor)
# model.build(input_shape=(1, 224, 224, 3))
input = Input((224, 224, 3))
model(input)
print(model.summary())


        


