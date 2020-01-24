from pathlib import Path


class ImageInfo:
    shape = width, height, channel = 224, 224, 3
    size = width, height


class ImagePath:
    image_path_dict = {'salad': r'/content/drive/My Drive/Image/salad.jpg',
                       'katsudon': r'/content/drive/My Drive/Image/katsudon.jpg'}


class DirectoryPath:
    weight = Path('weight')
