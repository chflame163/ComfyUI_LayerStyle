import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, mask2image
from .imagefunc import chop_image_v2, chop_mode_v2, shift_image, expand_mask



class DropShadowV3:

    def __init__(self):
        self.NODE_NAME = 'DropShadowV3'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_v2,),  # 混合模式
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),  # 透明度
                "distance_x": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1}),  # x_偏移
                "distance_y": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1}),  # y_偏移
                "grow": ("INT", {"default": 6, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 18, "min": 0, "max": 1000, "step": 1}),  # 模糊
                "shadow_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                "background_image": ("IMAGE", ),  #
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'drop_shadow_v2'
    CATEGORY = '😺dzNodes/LayerStyle'

    def drop_shadow_v2(self, layer_image, invert_mask, blend_mode, opacity,
                       distance_x, distance_y, grow, blur, shadow_color,
                       background_image=None, layer_mask=None
                       ):

        # If background image is empty, create transparent background image for each layer image
        if background_image == None:
            background_image = []
            for l in layer_image:
                m = tensor2pil(l)
                background_image.append(pil2tensor(Image.new('RGBA', (m.width, m.height), (0, 0, 0, 0))))

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        if len(l_masks) == 0:
            log(f"Error: {self.NODE_NAME} skipped, because the available mask is not found.", message_type='error')
            return (background_image,)

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        distance_x = -distance_x
        distance_y = -distance_y
        shadow_color = Image.new("RGBA", tensor2pil(l_images[0]).size, color=shadow_color)

        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            # preprocess
            _canvas = tensor2pil(background_image).convert('RGBA')
            _layer = tensor2pil(layer_image)

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {self.NODE_NAME} mask mismatch, dropped!", message_type='warning')

            if distance_x != 0 or distance_y != 0:
                __mask = shift_image(_mask, distance_x, distance_y)  # 位移
            shadow_mask = expand_mask(image2mask(__mask), grow, blur)  #扩张，模糊
            # 合成阴影
            alpha = tensor2pil(shadow_mask).convert('L')
            _shadow = chop_image_v2(_canvas, shadow_color, blend_mode, opacity)
            _canvas.paste(_shadow, mask=alpha)
            # 合成layer
            _canvas.paste(_layer, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerStyle: DropShadow V3": DropShadowV3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: DropShadow V3": "LayerStyle: DropShadow V3"
}