import torch
from .imagefunc import log, tensor2pil, pil2tensor, image_channel_split



class ImageChannelSplit:

    def __init__(self):
        self.NODE_NAME = 'ImageChannelSplit'

    @classmethod
    def INPUT_TYPES(self):
        channel_mode = ['RGBA', 'YCbCr', 'LAB', 'HSV']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (channel_mode,),  # 通道设置
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("channel_1", "channel_2", "channel_3", "channel_4",)
    FUNCTION = 'image_channel_split'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_channel_split(self, image, mode):

        c1_images = []
        c2_images = []
        c3_images = []
        c4_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i).convert('RGBA')
            channel1, channel2, channel3, channel4 = image_channel_split(_image, mode)
            c1_images.append(pil2tensor(channel1))
            c2_images.append(pil2tensor(channel2))
            c3_images.append(pil2tensor(channel3))
            c4_images.append(pil2tensor(channel4))

        log(f"{self.NODE_NAME} Processed {len(c1_images)} image(s).", message_type='finish')
        return (torch.cat(c1_images, dim=0), torch.cat(c2_images, dim=0), torch.cat(c3_images, dim=0), torch.cat(c4_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageChannelSplit": ImageChannelSplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageChannelSplit": "LayerUtility: ImageChannelSplit"
}