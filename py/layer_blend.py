from .imagefunc import *

class LayerBlend:

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
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'layer_blend'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def layer_blend(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  layer_mask=None,
                  ):

        #
        _layer = tensor2pil(layer_image)
        _canvas = tensor2pil(background_image)
        _mask = None
        if layer_mask is not None and invert_mask:
            layer_mask = 1 - layer_mask

        # 合成layer
        _comp = chop_image(tensor2pil(background_image), _layer, blend_mode, opacity)
        if layer_mask is not None:
            alpha = tensor2pil(layer_mask).convert('L')
            _canvas.paste(_comp, mask=alpha)
        else:
            _canvas.paste(_comp)

        ret_image = _canvas
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_ImageBlend": LayerBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_ImageBlend": "LayerStyle: ImageBlend"
}