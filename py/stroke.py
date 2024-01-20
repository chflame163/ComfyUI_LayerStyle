from .imagefunc import *

class Stroke:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        chop_mode = ['normal','multply','screen','add','subtract','difference','darker','lighter']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "stroke_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),  # 收缩值
                "stroke_width": ("INT", {"default": 8, "min": 0, "max": 999, "step": 1}),  # 扩张值
                "blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),  # 模糊
                "stroke_color": ("STRING", {"default": "#FF0000"}),  # 描边颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'stroke'
    CATEGORY = '😺dzNodes/LayerStyle'
    OUTPUT_NODE = True

    def stroke(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  stroke_grow, stroke_width, blur, stroke_color,
                  layer_mask=None
                  ):

        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image).convert('RGB')
        _mask = tensor2pil(layer_image).convert('RGBA').split()[-1]
        if layer_mask is not None:
            if invert_mask:
                layer_mask = 1 - layer_mask
            _mask = mask2image(layer_mask).convert('L')

        grow_offset = int(stroke_width / 2)
        inner_stroke = stroke_grow - grow_offset
        outer_stroke = inner_stroke + stroke_width
        inner_mask = expand_mask(image2mask(_mask), inner_stroke, blur)
        outer_mask = expand_mask(image2mask(_mask), outer_stroke, blur)
        strock_mask = subtract_mask(outer_mask, inner_mask)
        color_image = Image.new('RGB', size=_layer.size, color=stroke_color)
        blend_image = chop_image(_layer, color_image, blend_mode, opacity)
        _canvas.paste(_layer, mask=_mask)
        _canvas.paste(blend_image, mask=tensor2pil(strock_mask))
        ret_image = _canvas
        log('Stroke Processed.')
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle: Stroke": Stroke
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: Stroke": "LayerStyle: Stroke"
}