from .imagefunc import *

class ColorCorrectLAB:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "L": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "A": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "B": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LAB'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_correct_LAB(self, image, L, A, B):

        ret_images = []

        for image in image:

            _l, _a, _b = tensor2pil(image).convert('LAB').split()
            if L != 0 :
                _l = image_gray_offset(_l, L)
            if A != 0 :
                _a = image_gray_offset(_a, A)
            if B != 0 :
                _b = image_gray_offset(_b, B)
            ret_image = image_channel_merge((_l, _a, _b), 'LAB')

            ret_images.append(pil2tensor(ret_image))

        log(f'LAB Processed {len(ret_images)} image(s).')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: LAB": ColorCorrectLAB
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: LAB": "LayerColor: LAB"
}