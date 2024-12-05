import torch
import math
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import shift_image




class ChannelShake:

    def __init__(self):
        self.NODE_NAME = 'ChannelShake'


    @classmethod
    def INPUT_TYPES(self):
        channel_mode = ['RGB', 'RBG', 'BGR', 'BRG', 'GBR', 'GRB']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "distance": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 距离
                "angle": ("FLOAT", {"default": 40, "min": -360, "max": 360, "step": 0.1}),  # 角度
                "mode": (channel_mode,),  # 模式
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'channel_shake'
    CATEGORY = '😺dzNodes/LayerFilter'

    def channel_shake(self, image, distance, angle, mode, ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            R, G, B = _canvas.split()
            x = int(math.cos(angle) * distance)
            y = int(math.sin(angle) * distance)
            if mode.startswith('R'):
                R = shift_image(R.convert('RGB'), -x, -y).convert('L')
            if mode.startswith('G'):
                G = shift_image(G.convert('RGB'), -x, -y).convert('L')
            if mode.startswith('B'):
                B = shift_image(B.convert('RGB'), -x, -y).convert('L')
            if mode.endswith('R'):
                R = shift_image(R.convert('RGB'), x, y).convert('L')
            if mode.endswith('G'):
                G = shift_image(G.convert('RGB'), x, y).convert('L')
            if mode.endswith('B'):
                B = shift_image(B.convert('RGB'), x, y).convert('L')

            ret_image = Image.merge('RGB', [R, G, B])
            ret_images.append(pil2tensor(ret_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: ChannelShake": ChannelShake
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: ChannelShake": "LayerFilter: ChannelShake"
}