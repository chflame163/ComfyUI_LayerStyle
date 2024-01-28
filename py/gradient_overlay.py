from .imagefunc import *

class GradientOverlay:

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
                "start_color": ("STRING", {"default": "#FFBF30"}),  # 渐变开始颜色
                "start_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "end_color": ("STRING", {"default": "#FE0000"}),  # 渐变结束颜色
                "end_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "angle": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),  # 渐变角度
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'gradient_overlay'
    CATEGORY = '😺dzNodes/LayerStyle'
    OUTPUT_NODE = True

    def gradient_overlay(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  start_color, start_alpha, end_color, end_alpha, angle,
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

        _gradient = gradient(start_color, end_color, _layer.width, _layer.height, float(angle))

        # 合成layer
        _comp = chop_image(_layer, _gradient, blend_mode, opacity)
        if start_alpha < 255 or end_alpha < 255:
            #
            start_color = RGB_to_Hex((start_alpha, start_alpha, start_alpha))
            end_color = RGB_to_Hex((end_alpha, end_alpha, end_alpha))
            comp_alpha = gradint(start_color, end_color, _layer.width, _layer.height, float(angle))
            comp_alpha = ImageChops.invert(comp_alpha).convert('L')
            _comp.paste(_layer, comp_alpha)
        _canvas.paste(_comp, mask=_mask)
        ret_image = _canvas
        log('GradientOverlay Processed.')
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle: GradientOverlay": GradientOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: GradientOverlay": "LayerStyle: GradientOverlay"
}