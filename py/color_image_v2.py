from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, AnyType, load_custom_size



any = AnyType("*")

class ColorImageV2:

    def __init__(self):
        self.NODE_NAME = 'ColorImage V2'

    @classmethod
    def INPUT_TYPES(self):
        size_list = ['custom']
        size_list.extend(load_custom_size())
        return {
            "required": {
                "size": (size_list,),
                "custom_width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "custom_height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'color_image_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def color_image_v2(self, size, custom_width, custom_height, color, size_as=None ):

        if size_as is not None:
            if size_as.shape[0] > 0:
                _asimage = tensor2pil(size_as[0])
            else:
                _asimage = tensor2pil(size_as)
            width, height = _asimage.size
        else:
            if size == 'custom':
                width = custom_width
                height = custom_height
            else:
                try:
                    _s = size.split('x')
                    width = int(_s[0].strip())
                    height = int(_s[1].strip())
                except Exception as e:
                    log(f"Warning: {self.NODE_NAME} invalid size, check {custom_size_file}", message_type='warning')
                    width = custom_width
                    height = custom_height

        ret_image = Image.new('RGB', (width, height), color=color)
        return (pil2tensor(ret_image), )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ColorImage V2": ColorImageV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ColorImage V2": "LayerUtility: ColorImage V2"
}