from .imagefunc import *

colormap_list = ['autumn', 'bone', 'jet', 'winter', 'rainbow', 'ocean',
                 'summer', 'sprint', 'cool', 'HSV', 'pink', 'hot',
                 'parula', 'magma', 'inferno', 'plasma', 'viridis', 'cividis',
                 'twilight', 'twilight_shifted', 'turbo', 'deepgreen']

class ColorMap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "color_map": (colormap_list,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_map'
    CATEGORY = '😺dzNodes/LayerFilter'
    OUTPUT_NODE = True

    def color_map(self, image, color_map, opacity
                  ):

        _canvas = tensor2pil(image)
        _image = image_to_colormap(_canvas, colormap_list.index(color_map))
        ret_image = chop_image(_canvas, _image, 'normal', opacity)

        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: ColorMap": ColorMap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: ColorMap": "LayerFilter: ColorMap"
}