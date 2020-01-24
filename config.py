from pathlib import Path


class ImageInfo:
    shape = width, height, channel = 224, 224, 3
    size = width, height


gdrive_path = Path(r'/content/drive/My Drive')


class DirectoryPath:
    weight = gdrive_path/'weight'
    scored_param = gdrive_path/'scored_param'
    tfrecords = Path(r'/content/tfrecords')
    image = gdrive_path/'Image'


class ImagePath:
    image_path_dict = {image_name: str(DirectoryPath.image/f'{image_name}.jpg')
                       for image_name in ['salad', 'katsudon', 'farm', 'flower', 'watarfall']}
