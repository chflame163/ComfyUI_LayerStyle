import os
import glob
from .imagefunc import *

class ColorCorrectLUTapply:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "LUT": (LUT_LIST,),  # LUT文件
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LUTapply'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_correct_LUTapply(self, image, LUT):
        ret_images = []
        for image in image:

            _image = tensor2pil(image)
            lut_file = LUT_DICT[LUT]
            ret_image = lut_apply(_image, lut_file)
            ret_images.append(pil2tensor(ret_image))

        log(f'LUT Apply Processed {len(ret_images)} image(s).')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: LUT Apply": ColorCorrectLUTapply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: LUT Apply": "LayerColor: LUT Apply"
}