from pathlib import Path


class ImageInfo:
    shape = width, height, channel = 224, 224, 3
    size = width, height


class ImagePath:
    image_dir_path = Path(r'/content/drive/My Drive/Image')
    image_path_dict = {'salad': str(image_dir_path/'salad.jpg'),
                       'katsudon': str(image_dir_path/'katsudon.jpg')}


class DirectoryPath:
    weight = Path('weight')
