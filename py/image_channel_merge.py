from .imagefunc import *

class ImageChannelMerge:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        channel_mode = ['RGBA', 'YCbCr', 'LAB', 'HSV']
        return {
            "required": {
                "channel_1": ("IMAGE", ),  #
                "channel_2": ("IMAGE",),  #
                "channel_3": ("IMAGE",),  #
                "mode": (channel_mode,),  # 通道设置
            },
            "optional": {
                "channel_4": ("IMAGE",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_channel_merge'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def image_channel_merge(self, channel_1, channel_2, channel_3, mode, channel_4=None):

        _channel1 = tensor2pil(channel_1)
        _channel2 = tensor2pil(channel_2)
        _channel3 = tensor2pil(channel_3)
        _channel4 = Image.new('L', size=_channel1.size, color='white')
        if channel_4 is not None:
            _channel4 = tensor2pil(channel_4)
        ret_image = image_channel_merge((_channel1, _channel2, _channel3, _channel4), mode)
        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageChannelMerge": ImageChannelMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageChannelMerge": "LayerUtility: ImageChannelMerge"
}