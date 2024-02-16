from .imagefunc import *

NODE_NAME = 'LUT Apply'

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
        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)
            lut_file = LUT_DICT[LUT]
            ret_image = lut_apply(_image, lut_file)
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: LUT Apply": ColorCorrectLUTapply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: LUT Apply": "LayerColor: LUT Apply"
}