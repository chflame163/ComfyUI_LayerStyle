from .imagefunc import *

class GetImageSize:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = 'get_image_size'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def get_image_size(self, image,):

        _image = tensor2pil(image).convert('RGB')

        return (_image.width, _image.height,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: GetImageSize": GetImageSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: GetImageSize": "LayerUtility: GetImageSize"
}