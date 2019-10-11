import numpy as np
from PIL import Image
import tensorflow as tf


def get_dataset(img_path, length):
    return tf.data.Dataset.from_generator(
        Digits2D(img_path, length=length).generate,
        output_types=(float),
        output_shapes=(2, ),
    )


class Digits2D:
    def __init__(self, img_path, length=100, buff_size=10000):
        img = np.array(Image.open(img_path))
        self.img = img / 255
        self.counter = 0
        self.length = length
        self.buff_size = buff_size
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = np.random.rand(self.buff_size, 2)

    def generate(self):
        img = self.img
        h, w = img.shape
        while True:
            for x, y in self.buffer:
                rx = (w * x - 0.5).round().astype(int)
                ry = (h * (1 - y) - 0.5).round().astype(int)

                if img[ry, rx] > 0.3:
                    self.counter += 1
                    yield (6 * (x - 0.5), 6 * (y - 0.5))

                if self.counter >= self.length:
                    return

            self.reset_buffer()