from .imagefunc import *

NODE_NAME = 'LUT Apply'

class ColorCorrectLUTapply:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        (LUT_DICT, _) = get_resource_dir()
        LUT_LIST = list(LUT_DICT.keys())

        color_space_list = ['linear', 'log']

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "LUT": (LUT_LIST,),  # LUT文件
                "color_space":  (color_space_list,),
                "strength": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LUTapply'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_LUTapply(self, image, LUT, color_space, strength):

        (LUT_DICT, _) = get_resource_dir()
        log(f"LUT_DICT={LUT_DICT}")
        ret_images = []
        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)

            lut_file = LUT_DICT[LUT]
            ret_image = apply_lut(_image, lut_file=lut_file, colorspace=color_space, strength=strength)

            if _image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, _image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)




NODE_CLASS_MAPPINGS = {
    "LayerColor: LUT Apply": ColorCorrectLUTapply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: LUT Apply": "LayerColor: LUT Apply"
}