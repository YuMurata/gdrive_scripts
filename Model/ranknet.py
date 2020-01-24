import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
from .grad_cam import GradCam
from .evaluate_network import build_evaluate_network

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import psutil

layers = tf.keras.layers


class RankNet:
    SCOPE = 'predict_model'
    PREDICTABLE_MODEL_FILE_NAME = 'predictable_model.h5'
    TRAINABLE_MODEL_FILE_NAME = 'trainable_model.h5'

    def __init__(self, image_shape: tuple, *, use_vgg16: bool = True):
        self.image_shape = image_shape
        width, height = image_shape[0], image_shape[1]

        with tf.name_scope(RankNet.SCOPE):
            evaluate_network = build_evaluate_network(
                image_shape, use_vgg16=use_vgg16)
            self.grad_cam = GradCam(evaluate_network, (width, height))

            left_input = tf.keras.Input(shape=image_shape)
            right_input = tf.keras.Input(shape=image_shape)

            left_output = evaluate_network(left_input)
            right_output = evaluate_network(right_input)

            concated_output = layers.Concatenate()([left_output, right_output])

            with tf.name_scope('predictable_model'):
                self.predictable_model = tf.keras.Model(inputs=left_input,
                                                        outputs=left_output)
            with tf.name_scope('trainable_model'):
                self.trainable_model = tf.keras.Model(inputs=[left_input,
                                                              right_input],
                                                      outputs=concated_output)

            loss = \
                tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)
            self.trainable_model.compile(optimizer='adam', loss=loss)

        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.g_drive = GoogleDrive(gauth)

    def train(self, dataset: tf.data.Dataset, *, log_dir_path: str,
              valid_dataset: tf.data.Dataset, epochs=10, steps_per_epoch=30):
        callbacks = tf.keras.callbacks

        cb = []

        weights_dir_path = Path(log_dir_path)/'weights'
        weights_dir_path.mkdir(parents=True, exist_ok=True)

        # cb.append(callbacks.EarlyStopping())
        cb.append(tf.keras.callbacks.ModelCheckpoint(
            str(weights_dir_path/'{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5'), save_weights_only=True, monitor='val_loss', save_best_only=True))

        if log_dir_path is not None:
            cb.append(callbacks.TensorBoard(log_dir=log_dir_path,
                                            write_graph=True))

        def remove_old_weights():
            dsk = psutil.disk_usage('/content/drive')
            if dsk.percent > 80:
                for weights_dir_info in self.g_drive.ListFile({'q': 'title = "weights"'}).GetList():
                    weights_dir_id = weights_dir_info['id']
                    weights_list = self.g_drive.ListFile(
                        {'q': f'"{weights_dir_id}" in parents'}).GetList()
                    weights_list.sort(key=lambda weights: weights['title'])
                    for weights in weights_list[:-1]:
                        weights.Delete()

        cb.append(tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: remove_old_weights()))

        self.trainable_model.fit(dataset, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=cb, validation_data=valid_dataset,
                                 validation_steps=10)

    def save(self, save_dir_path: str):
        save_dir_path = Path(save_dir_path)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        self.trainable_model.save_weights(
            str(Path(save_dir_path) /
                RankNet.TRAINABLE_MODEL_FILE_NAME))

    def load(self, load_file_path: str):
        self.trainable_model.load_weights(load_file_path)

    def save_model_structure(self, save_dir_path: str):
        save_dir_path = Path(save_dir_path)
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True)

        tf.keras.utils.plot_model(self.predictable_model,
                                  str(save_dir_path/'predictable_model.png'),
                                  show_shapes=True)

        tf.keras.utils.plot_model(self.trainable_model,
                                  str(save_dir_path/'trainable_model.png'),
                                  show_shapes=True)

    def _image_to_array(self, image: Image.Image):
        width, height = self.image_shape[0], self.image_shape[1]
        resized_image = image.resize((width, height))
        return np.asarray(resized_image).astype(np.float32)/255

    def predict(self, data_list: list):
        image_array_list = np.array([self._image_to_array(data['image'])
                                     for data in data_list])

        return self.predictable_model.predict(image_array_list)


if __name__ == '__main__':
    model = RankNet()
    model.load(r'C:\Users\init\Documents\PythonScripts\EnhanceImageFromUserPreference\Experiment\Questionnaire\summary\test\katsudon\1118\1909')

    import matplotlib.pyplot as plt

    image_path_list = [
        r'C:\Users\init\Documents\PythonScripts\EnhanceImageFromUserPreference\Experiment\Questionnaire\image\katsudon\1\1.jpg',
        r'C:\Users\init\Documents\PythonScripts\EnhanceImageFromUserPreference\Experiment\Questionnaire\image\katsudon\2\2.png',
        r'C:\Users\init\Documents\PythonScripts\EnhanceImageFromUserPreference\Experiment\Questionnaire\image\salad\1\1.jpg',
        r'C:\Users\init\Documents\PythonScripts\EnhanceImageFromUserPreference\Experiment\Questionnaire\image\salad\2\2.jpg',
    ]

    for image_path in image_path_list:
        image = Image.open(image_path).convert('RGB')
        cam = model.grad_cam.get_cam(np.array(image))

        plt.figure()
        plt.imshow(cam)

    plt.show()
