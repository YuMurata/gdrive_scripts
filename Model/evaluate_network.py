import tensorflow as tf


def _build_my_cnn(input_shape: tuple):
    layers = tf.keras.layers

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=5,
                            padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(filters=64, kernel_size=5,
                            padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    return model


def _build_vgg16(input_shape: tuple):
    vgg16 = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in vgg16.layers[:15]:
        layer.trainable = False

    return vgg16


def build_evaluate_network(input_shape: tuple, *, use_vgg16: bool = True) -> tf.keras.Sequential:
    convolution_layer = _build_vgg16(
        input_shape) if use_vgg16 else _build_my_cnn(input_shape)

    top_layer = tf.keras.Sequential()
    top_layer.add(tf.keras.layers.Flatten(
        input_shape=convolution_layer.output_shape[1:]))
    top_layer.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    top_layer.add(tf.keras.layers.Dropout(rate=0.5))
    top_layer.add(tf.keras.layers.Dense(units=1))

    model = tf.keras.Model(inputs=convolution_layer.input,
                           outputs=top_layer(convolution_layer.output))
    return model
