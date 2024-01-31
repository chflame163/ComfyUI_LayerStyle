from .imagefunc import *

class RestoreCropBox:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "crop_box": ("BOX",),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box,
                         croped_mask=None
                         ):

        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(croped_image).convert('RGB')
        _mask = Image.new('L', size=_layer.size, color='white')
        if croped_mask is not None:
            if invert_mask:
                croped_mask = 1 - croped_mask
            _mask = mask2image(croped_mask).convert('L')
        ret_mask = Image.new('L', size=_canvas.size, color='black')
        _canvas.paste(_layer, box=tuple(crop_box), mask=_mask)
        ret_mask.paste(_mask, box=tuple(crop_box))

        return (pil2tensor(_canvas), image2mask(ret_mask),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: RestoreCropBox": RestoreCropBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RestoreCropBox": "LayerUtility: RestoreCropBox"
}