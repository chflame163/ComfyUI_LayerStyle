from .imagefunc import *

class MaskEdgeShrink:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "shrink_level": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1}),
                "soft": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "edge_shrink": ("INT", {"default": 1, "min": 0, "max": 999, "step": 1}),
                "edge_reserve": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_edge_shrink'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_edge_shrink(self, mask, invert_mask, shrink_level, soft, edge_shrink, edge_reserve):

        glow_range = shrink_level * soft
        blur = 12
        _inputmask = mask2image(mask)
        _canvas = Image.new('RGB', size=_inputmask.size, color='black')
        _layer = Image.new('RGB', size=_inputmask.size, color='white')

        if invert_mask:
            mask = 1 - mask
        _mask = mask2image(mask).convert('L')

        loop_grow = glow_range
        inner_mask = _mask
        for x in range(shrink_level):

            _color = step_color('#FFFFFF', '#000000', shrink_level, x)
            glow_mask = expand_mask(image2mask(inner_mask), -loop_grow, blur / (x+0.1))  #扩张，模糊
            # 合成
            color_image = Image.new("RGB", _layer.size, color=_color)
            alpha = tensor2pil(mask_invert(glow_mask)).convert('L')
            _glow = chop_image(_layer, color_image, 'subtract', int(step_value(1, 100, shrink_level, x)))
            _layer.paste(_glow, mask=alpha)
            loop_grow = loop_grow - int(glow_range / shrink_level)
        # 合成layer
        _edge = tensor2pil(expand_mask(mask, -edge_shrink, 0)).convert('RGB')
        _layer = chop_image(_layer, _edge, 'normal', edge_reserve)
        _layer.paste(_canvas, mask=ImageChops.invert(_mask))
        ret_mask = image2mask(_layer)

        return (ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskEdgeShrink": MaskEdgeShrink
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskEdgeShrink": "LayerMask: MaskEdgeShrink"
}