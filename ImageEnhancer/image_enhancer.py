from PIL import Image
from ImageEnhancer import enhance_dict
import typing


class ImageEnhancerException(Exception):
    pass


class ImageEnhancer:
    def __init__(self, image_path: str):
        self.org_image = Image.open(image_path).convert('RGB')
        if self.org_image is None:
            raise ImageEnhancerException('image not found')

        self.org_width, self.org_height = self.org_image.size

    def _enhance(self, image_parameter: dict, image: Image.Image) \
            -> Image.Image:

        ret_image = image
        for enhance_name, enhance_class in enhance_dict.items():
            if enhance_name not in image_parameter:
                continue

            ret_image = enhance_class(ret_image).enhance(
                image_parameter[enhance_name])
        return ret_image

    def enhance(self, image_parameter: dict) -> Image.Image:
        return self._enhance(image_parameter, self.org_image)


class ResizableEnhancer(ImageEnhancer):
    def __init__(self, image_path: str, resized_size: typing.Tuple[int, int]):
        super(ResizableEnhancer, self).__init__(image_path)
        self.resize_image = self.org_image.resize(resized_size)

    def resized_enhance(self, image_parameter: dict) -> Image.Image:
        return self._enhance(image_parameter, self.resize_image)
