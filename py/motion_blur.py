from .imagefunc import *

class MotionBlur:

    def __init__(self):
        pass

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
    OUTPUT_NODE = True

    def motion_blur(self, image, angle, blur):

        ret_images = []

        for image in image:

            _canvas = tensor2pil(image).convert('RGB')
            ret_image = motion_blur(_canvas, angle, blur)

            ret_images.append(pil2tensor(ret_image))

        log(f'MotionBlur Processed {len(ret_images)} image(s).')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: MotionBlur": MotionBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: MotionBlur": "LayerFilter: MotionBlur"
}