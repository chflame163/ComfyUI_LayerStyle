import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image_watercolor, chop_image



class WaterColor:

    def __init__(self):
        self.NODE_NAME = 'WaterColor'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "line_density": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),  # 透明度
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'water_color'
    CATEGORY = '😺dzNodes/LayerFilter'

    def water_color(self, image, line_density, opacity
                  ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            _image = image_watercolor(_canvas, level=101-line_density)
            ret_image = chop_image(_canvas, _image, 'normal', opacity)

            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: WaterColor": WaterColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: WaterColor": "LayerFilter: WaterColor"
}