import numpy as np
import tensorflow as tf
from .generate_param import generate_random_param
from tqdm import trange
from ImageEnhancer import ResizableEnhancer
from .evaluator import Evaluator


class Writer:
    def __init__(self, save_file_path: str):
        self.writer = tf.io.TFRecordWriter(save_file_path)

    def write(self, left_array: np.array, right_array: np.array, label: int):
        features = \
            tf.train.Features(
                feature={
                    'label':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])),
                    'left_image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[left_array.tobytes()])),
                    'right_image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[right_array.tobytes()]))
                }
            )

        example = tf.train.Example(features=features)
        record = example.SerializeToString()
        self.writer.write(record)


def make_tfrecords(save_file_path: str, generate_num: int, enhancer: ResizableEnhancer, evaluator: Evaluator):
    writer = Writer(save_file_path)
    for _ in trange(generate_num, desc='write tfrecords'):
        left_param = generate_random_param()
        right_param = generate_random_param()

        left_array = np.array(enhancer.resized_enhance(left_param))
        right_array = np.array(enhancer.resized_enhance(right_param))

        left_score = evaluator.evaluate(left_param)
        right_score = evaluator.evaluate(right_param)

        label = 0 if left_score > right_score else 1

        writer.write(left_array, right_array, label)
