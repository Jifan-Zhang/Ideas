import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import numpy as np

class Model_builder:
    """
    A model layers wrapper, building layer blocks.
    The instance variable remembers the layer structure.
        input_shape: The image shape CNN trains on
    """
    def __init__(self, input_shape = (1024,1024,3), output_shape = (512,512,3)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.blocks = []

    def __bottle_neck_check__(f):
        def inner(self, *args, **kwargs):
            f(self, *args, **kwargs)
            if(self.__dict__.get("Discriminator")): # Only painter checks bottle neck
                x = self.blocks[-1]
                if(x.get_shape()[1]<self.output_shape[0] or x.get_shape()[2]<self.output_shape[1]):
                    raise RuntimeError(f"The model has formed a bottle neck structure {(x.get_shape()[1],x.get_shape()[2])} < {(self.output_shape[0], self.output_shape[1])}, which should be recovered with up sampling, which is not implemented in the version.")
        return inner

    def input(self):
        out = tf.keras.Input(shape=self.input_shape)
        self.blocks.append(out)
        """"""
        if(self.__dict__.get("Discriminator") is None): # Discriminator scales input (Only painter stores discriminator in its instance)
            out = out/255.
            self.blocks.append(out)
        return out

    @__bottle_neck_check__
    def conv_block(self, n_filter=5, filter_size=(3,3), padding = "valid", strides=1):
        x = self.blocks[-1]
        out = tf.keras.layers.Conv2D(n_filter, filter_size, strides=strides, padding = padding, activation="selu")(x)
        self.blocks.append(out)
        return out

    @__bottle_neck_check__
    def pooling_block(self, strides=(2,2)):
        x = self.blocks[-1]
        out = tf.keras.layers.MaxPool2D()(x)
        self.blocks.append(out)
        return out

    @__bottle_neck_check__
    def conv_pool_block(self, n_filter=5, filter_size=(3,3), strides=1):
        x = self.blocks[-1]
        x = tf.keras.layers.Conv2D(n_filter, filter_size, strides=strides, padding = "same",activation="selu")(x)
        out = tf.keras.layers.MaxPool2D()(x)
        self.blocks.append(out)
        return out

    def fully_connected(self,n):
        x = self.blocks[-1]
        if(len(x.get_shape())!=2):
            x = tf.keras.layers.Flatten()(x)
        if(n==2):
            activation = "softmax"
        elif(n==1):
            activation = "linear"
        else:
            activation = "selu"
        out = tf.keras.layers.Dense(n, activation=activation)(x)
        self.blocks.append(out)
        return out

    def top_block(self):
        x = self.blocks[-1]
        width = x.get_shape()[1]
        height = x.get_shape()[2]
        f_width = width + 1 - self.output_shape[0]
        f_height = height + 1 - self.output_shape[1]
        out = tf.keras.layers.Conv2D(3, (f_width,f_height) ,padding="valid", activation = "selu")(x)
        self.blocks.append(out)
        return out
    
