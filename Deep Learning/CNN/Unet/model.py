import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Activation, UpSampling2D

def conv_block(filters, x):
    x = Conv2D(filters=filters, 
                kernel_size=(3,3), 
                padding='same', 
                activation='relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=filters, 
                kernel_size=(3,3), 
                padding='same', 
                activation='relu')(x)
    out = BatchNormalization()(x)
    return out 


def down_sample(filters, x):
    x = MaxPool2D((2,2))(x)
    out = conv_block(filters, x)
    return out


def up_sample(filters, x, x1):
    x = Conv2D(filters=filters,
                kernel_size=(2,2),
                padding='same',
                activation='relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x1, x])
    out = conv_block(filters, x)
    return out

def Unet():

    # Encoder
    inputs = Input(shape=(224, 224, 3), name="input_image")
    down1 = conv_block(64, inputs)
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
                padding='same',
                activation='sigmoid')(up4)

    model = Model(inputs, out)
    return model

    def summary(self):
        x = tf.keras.layers.Input(shape=(224, 224, 3))
        return Model(inputs=[x], outputs=self.call(x))



def main():
    model = Unet()
    model.summary()


if __name__ == '__main__':
    main()



        


