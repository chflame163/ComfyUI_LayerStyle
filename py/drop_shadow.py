from .imagefunc import *

class DropShadow:

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
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),  # 透明度
                "distance_x": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1}),  # x_偏移
                "distance_y": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1}),  # y_偏移
                "grow": ("INT", {"default": 6, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 18, "min": 0, "max": 100, "step": 1}),  # 模糊
                "shadow_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'drop_shadow'
    CATEGORY = '😺dzNodes/LayerStyle'
    OUTPUT_NODE = True

    def drop_shadow(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity, distance_x, distance_y,
                  grow, blur, shadow_color,
                  layer_mask=None
                  ):

        # preprocess
        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image)
        if _layer.mode == 'RGBA':
            _mask = tensor2pil(layer_image).convert('RGBA').split()[-1]
        else:
            _mask = Image.new('L', _layer.size, 'white')
        _layer = _layer.convert('RGB')
        if layer_mask is not None:
            if invert_mask:
                layer_mask = 1 - layer_mask
            _mask = mask2image(layer_mask).convert('L')
        if _mask.size != _layer.size:
            _mask = Image.new('L', _layer.size, 'white')
            log('Warning: mask mismatch, droped!')

        distance_x = -distance_x
        distance_y = -distance_y
        if distance_x != 0 or distance_y != 0:
            __mask = shift_image(_mask, distance_x, distance_y)  # 位移
        shadow_mask = expand_mask(image2mask(__mask), grow, blur)  #扩张，模糊
        # 合成阴影
        shadow_color = Image.new("RGB", _layer.size, color=shadow_color)
        alpha = tensor2pil(shadow_mask).convert('L')
        _shadow = chop_image(_canvas, shadow_color, blend_mode, opacity)
        _canvas.paste(_shadow, mask=alpha)
        # 合成layer
        _canvas.paste(_layer, mask=_mask)
        ret_image = _canvas
        log('DropShadow Processed.')
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle: DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: DropShadow": "LayerStyle: DropShadow"
}