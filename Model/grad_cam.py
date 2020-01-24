import tensorflow as tf
import numpy as np
import cv2


class GradCam:
    def __init__(self, model: tf.keras.Model, input_image_size: tuple):
        self.model = model
        self.input_image_size = input_image_size

        # self.model = tf.keras.applications.VGG16(
        #     weights='imagenet')
        # self.input_image_size = 224, 224

    def get_cam(self, image_array: np.array, layer_name: str):
        grad_model = tf.keras.Model(inputs=[self.model.inputs],
                                    outputs=[self.model.get_layer(layer_name).output,
                                             self.model.output])

        input_image_array = np.array(
            [cv2.resize(image_array, self.input_image_size).astype(np.float32)/255])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(
                [input_image_array, input_image_array])

        grads = tape.gradient(predictions, conv_outputs)[0]
        output = conv_outputs[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = np.dot(output, weights)

        # print('output: ', output)
        # print('weights: ', weights)

        height, width, _ = image_array.shape
        # Process CAM
        cam = cv2.resize(cam, (width, height), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)

        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        alpha = 0.5
        cam = cv2.addWeighted(image_array, alpha, cam, 1-alpha, 0)
        return cam
