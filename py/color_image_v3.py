from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, AnyType, load_custom_size
from .color_name import LS_ColorName


any = AnyType("*")

class LS_ColorImageV3:

    def __init__(self):
        self.NODE_NAME = 'ColorImage V3'

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
                "color_name": ("STRING", {"default": "white",},),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "color",)
    FUNCTION = 'color_image_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def color_image_v2(self, size, custom_width, custom_height, color, size_as=None, color_name=None):

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
                    log(f'Warning: {self.NODE_NAME} invalid size, check "custom_size.ini"', message_type='warning')
                    width = custom_width
                    height = custom_height

        if color_name is not None:
            try:
                color_table = LS_ColorName()
                color = color_table.XKCD_NAME_TO_HEX[color_name]
            except KeyError:
                log(f"{self.NODE_NAME}: {color_name} not in XKCD color table, use custom color value.")
        ret_image = Image.new('RGB', (width, height), color=color)
        return (pil2tensor(ret_image), color,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ColorImage V3": LS_ColorImageV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ColorImage V3": "LayerUtility: ColorImage V3"
}