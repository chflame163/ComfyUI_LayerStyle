from .imagefunc import *

class GradientOverlay:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

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

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []

        for b in background_image:
            b_images.append(b)
        for l in layer_image:
            l_images.append(l)
            m = tensor2pil(l)
            if tensor2pil(l).mode == 'RGBA':
                l_masks.append(m.convert('RGBA').split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if layer_mask is not None:
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(m).convert('L'))
        max_batch = max(len(b_images), len(l_images), len(l_masks))
        width, height =  tensor2pil(l_images[0]).size

        _gradient = gradient(start_color, end_color, width, height, float(angle))
        #
        start_color = RGB_to_Hex((start_alpha, start_alpha, start_alpha))
        end_color = RGB_to_Hex((end_alpha, end_alpha, end_alpha))
        comp_alpha = gradient(start_color, end_color, width, height, float(angle))
        comp_alpha = ImageChops.invert(comp_alpha).convert('L')
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            # preprocess
            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log('Warning: mask mismatch, droped!')

            # 合成layer
            _comp = chop_image(_layer, _gradient, blend_mode, opacity)
            if start_alpha < 255 or end_alpha < 255:
                _comp.paste(_layer, comp_alpha)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f'GradientOverlay Processed {len(ret_images)} image(s).')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerStyle: GradientOverlay": GradientOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: GradientOverlay": "LayerStyle: GradientOverlay"
}