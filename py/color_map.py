import torch
from .imagefunc import log, tensor2pil, pil2tensor, chop_image
from .imagefunc import image_to_colormap



colormap_list = ['autumn', 'bone', 'jet', 'winter', 'rainbow', 'ocean',
                 'summer', 'sprint', 'cool', 'HSV', 'pink', 'hot',
                 'parula', 'magma', 'inferno', 'plasma', 'viridis', 'cividis',
                 'twilight', 'twilight_shifted', 'turbo', 'deepgreen']

class ColorMap:

    def __init__(self):
        self.NODE_NAME = 'ColorMap'

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

    def color_map(self, image, color_map, opacity
                  ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i)
            _image = image_to_colormap(_canvas, colormap_list.index(color_map))
            ret_image = chop_image(_canvas, _image, 'normal', opacity)

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: ColorMap": ColorMap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: ColorMap": "LayerFilter: ColorMap"
}