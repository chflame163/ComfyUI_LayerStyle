import torch
from .imagefunc import tensor2pil

class GetImageSize:

    def __init__(self):
        self.NODE_NAME = 'GetImageSize'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOX")
    RETURN_NAMES = ("width", "height",  "original_size")
    FUNCTION = 'get_image_size'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_image_size(self, image,):

        if image.shape[0] > 0:
            image = torch.unsqueeze(image[0], 0)
        _image = tensor2pil(image)

        return (_image.width, _image.height, [_image.width, _image.height],)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GetImageSize": GetImageSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetImageSize": "LayerUtility: GetImageSize"
}