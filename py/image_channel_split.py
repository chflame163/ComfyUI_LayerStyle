from .imagefunc import *

class ImageChannelSplit:

    def __init__(self):
        pass

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
    OUTPUT_NODE = True

    def image_channel_split(self, image, mode):

        _image = tensor2pil(image).convert('RGBA')
        channel1, channel2, channel3, channel4 = image_channel_split(_image, mode)

        return (pil2tensor(channel1), pil2tensor(channel2), pil2tensor(channel3), pil2tensor(channel4),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageChannelSplit": ImageChannelSplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageChannelSplit": "LayerUtility: ImageChannelSplit"
}