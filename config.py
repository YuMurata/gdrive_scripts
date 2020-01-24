from pathlib import Path


class ImageInfo:
    shape = width, height, channel = 224, 224, 3
    size = width, height


class ImagePath:
    image_path_dict = {'salad': r'Image/salad.jpg',
                       'katsudon': r'Image/katsudon.jpg'}


class DirectoryPath:
    weight = Path('weight')
