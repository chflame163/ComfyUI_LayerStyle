from .imagefunc import *

class ExtendCanvas:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "top": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "color": ("COLOR", {"default": "#000000"},),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'extend_canvas'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def extend_canvas(self, image, invert_mask,
                  top, bottom, left, right, color,
                  mask=None,
                  ):

        _image = tensor2pil(image).convert('RGB')
        _mask = tensor2pil(image).convert('RGBA').split()[-1]
        if mask is not None:
            if invert_mask:
                mask = 1 - mask
            _mask = mask2image(mask).convert('L')

        width = _image.width + left + right
        height = _image.height + top + bottom
        _canvas = Image.new('RGB', (width, height), color)
        _mask_canvas = Image.new('L', (width, height), "black")

        _canvas.paste(_image, box=(left,top))
        _mask_canvas.paste(_mask.convert('L'), box=(left, top))
        ret_image = _canvas
        ret_mask = image2mask(_mask_canvas)
        log('ExtendCanvas Processed.')
        return (pil2tensor(ret_image), ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ExtendCanvas": ExtendCanvas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ExtendCanvas": "LayerUtility: ExtendCanvas"
}