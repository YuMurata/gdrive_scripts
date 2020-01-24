from pathlib import Path


class ImageInfo:
    shape = width, height, channel = 224, 224, 3
    size = width, height


gdrive_path = Path(r'/content/drive/My Drive')


class ImagePath:
    jpg_image_name_list = ['salad', 'katsudon',
                           'farm', 'flower', 'watarfall']
    image_dir_path = gdrive_path/'Image'
    image_path_dict = {image_name: str(image_dir_path/f'{image_name}.jpg')
                       for image_name in jpg_image_name_list}


class DirectoryPath:
    weight = gdrive_path/'weight'
    scored_param = gdrive_path/'scored_param'
    tfrecords = Path(r'/content/tfrecords')
