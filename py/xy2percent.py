
from .imagefunc import tensor2pil

class XYtoPercent:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("x_percent", "y_percent",)
    FUNCTION = 'xy_to_percent'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def xy_to_percent(self, background_image, layer_image, x, y,):

        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image).convert('RGB')
        x_percent = (x + _layer.width / 2) / _canvas.width * 100.0
        y_percent = (y + _layer.height / 2) / _canvas.height * 100.0

        return (x_percent, y_percent,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: XY to Percent": XYtoPercent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: XY to Percent": "LayerUtility: XY to Percent"
}