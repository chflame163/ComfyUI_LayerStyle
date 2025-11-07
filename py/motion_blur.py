import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, motion_blur



class MotionBlur:

    def __init__(self):
        self.NODE_NAME = 'MotionBlur'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "angle": ("INT", {"default": 0, "min": -90, "max": 90, "step": 1}),  # 角度
                "blur": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'motion_blur'
    CATEGORY = '😺dzNodes/LayerFilter'

    def motion_blur(self, image, angle, blur):

        ret_images = []

        for i in image:

            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')

            ret_images.append(pil2tensor(motion_blur(_canvas, angle, blur)))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: MotionBlur": MotionBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: MotionBlur": "LayerFilter: MotionBlur"
}