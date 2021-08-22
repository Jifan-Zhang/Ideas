import tensorflow as tf
import numpy as np
from PIL import Image
from collections.abc import Iterator

class Generate(Iterator):
    """
    Infinitely generate color-gradient image (numpy array) for training.
        path: absolute path to the input RGB image
        out_shape: training image (1, width, height, channels)
    """
    def __init__(self, path, out_shape=(1024,1024)):
        self.path = path
        self.out_shape = out_shape
        self.input = np.expand_dims(np.array(Image.open(path)), axis=0)
        self.input_shape = self.input.shape

    def __next__(self):
        x = np.random.randint(0, self.input_shape[1] - self.out_shape[0])
        y = np.random.randint(0, self.input_shape[2] - self.out_shape[1])
        return self.input[:, x:x+self.out_shape[0], y:y+self.out_shape[1], :]
