import math

from .imagefunc import *

class ImageShift:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "shift_x": ("INT", {"default": 256, "min": -9999, "max": 9999, "step": 1}),
                "shift_y": ("INT", {"default": 256, "min": -9999, "max": 9999, "step": 1}),
                "cyclic": ("BOOLEAN", {"default": True}),  # 反转mask#
                "background_color": ("STRING", {"default": "#000000"}),
                "border_mask_width": ("INT", {"default": 20, "min": 0, "max": 999, "step": 1}),
                "border_mask_blur": ("INT", {"default": 12, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("image", "mask", "border_mask")
    FUNCTION = 'image_shift'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def image_shift(self, image, shift_x, shift_y,
                    cyclic, background_color,
                    border_mask_width, border_mask_blur,
                    mask=None
                    ):

        ret_images = []
        ret_masks = []
        ret_border_masks = []
        for image in image:

            shift_x, shift_y = -shift_x, -shift_y
            _canvas = tensor2pil(image).convert('RGB')
            _mask = tensor2pil(image).convert('RGBA').split()[-1]
            if mask is not None:
                _mask = mask2image(mask).convert('L')
            _border =  Image.new('L', size=_canvas.size, color='black')
            _border = draw_border(_border, border_width=border_mask_width, color='#FFFFFF')
            _border = _border.resize(_canvas.size)
            _canvas = shift_image(_canvas, shift_x, shift_y, background_color=background_color, cyclic=cyclic)
            _mask = shift_image(_mask, shift_x, shift_y, background_color='#000000', cyclic=cyclic)
            _border = shift_image(_border, shift_x, shift_y, background_color='#000000', cyclic=cyclic)
            _border = gaussian_blur(_border, border_mask_blur)

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(_mask))
            ret_border_masks.append(image2mask(_border))

        log(f'ImageShift Processed {len(ret_images)} image(s).')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), torch.cat(ret_border_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageShift": ImageShift
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageShift": "LayerUtility: ImageShift"
}