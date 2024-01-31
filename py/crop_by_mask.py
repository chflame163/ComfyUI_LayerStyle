from .imagefunc import *

class CropByMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['min_bounding_rect', 'max_inscribed_rect']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask_for_crop": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def crop_by_mask(self, image, mask_for_crop, invert_mask, detect,
                  top_reserve, bottom_reserve, left_reserve, right_reserve
                  ):

        _canvas = tensor2pil(image).convert('RGB')
        if invert_mask:
            mask_for_crop = 1 - mask_for_crop
        _mask = mask2image(mask_for_crop)
        bluredmask = gaussian_blur(_mask, 20).convert('L')
        x = 0
        y = 0
        width = 0
        height = 0
        if detect == "min_bounding_rect":
            (x, y, width, height) = min_bounding_rect(bluredmask)
        if detect == "max_inscribed_rect":
            (x, y, width, height) = max_inscribed_rect(bluredmask)

        x1 = x - left_reserve if x - left_reserve > 0 else 0
        y1 = y - top_reserve if y - top_reserve > 0 else 0
        x2 = x + width + right_reserve if x + width + right_reserve < _canvas.width else _canvas.width
        y2 = y + height + bottom_reserve if y + height + bottom_reserve < _canvas.height else _canvas.height
        preview_image = tensor2pil(mask_for_crop).convert('RGB')
        preview_image = draw_rect(preview_image, x, y, width, height, line_color="#F00000", line_width=(width+height)//100)
        preview_image = draw_rect(preview_image, x1, y1, x2 - x1, y2 - y1,
                                  line_color="#00F000", line_width=(width+height)//200)
        crop_box = (x1, y1, x2, y2)
        ret_image = _canvas.crop(crop_box)
        ret_mask = _mask.crop(crop_box)

        return (pil2tensor(ret_image), image2mask(ret_mask), list(crop_box), pil2tensor(preview_image),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropByMask": CropByMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask": "LayerUtility: CropByMask"
}