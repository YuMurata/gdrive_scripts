from submodule import ImageRankNet
from config import ImageInfo
import tensorflow as tf


class MyCNN(ImageRankNet.EvaluateBody):
    def __init__(self):
        super().__init__(ImageInfo.shape)

    def build(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                         padding='same', activation='relu',
                                         input_shape=self.image_shape))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                                         padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        return model


class Xception(ImageRankNet.EvaluateBody):
    def __init__(self):
        super().__init__(ImageInfo.shape)

    def build(self):
        model = \
            tf.keras.applications.xception.Xception(
                include_top=False,
                weights='imagenet',
                input_shape=self.image_shape,
                pooling='avg')

        for layer in model.layers[:-50]:
            layer.trainable = False

        return model
