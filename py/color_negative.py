import torch
from .imagefunc import log, tensor2pil, pil2tensor
from .imagefunc import image_channel_merge, RGB2RGBA




negative_channel_list = ["RGB", "Mono", "R", "G", "B",]


def invert_specific_channel(image: torch.Tensor, channels_to_invert: list) -> torch.Tensor:
    """
    对图像中特定的通道进行颜色取反，其余通道保持不变。

    参数:
        image (Tensor): 形状为 (B, H, W, 3)，值在 [0.0, 1.0] 的 float 类型张量。
        channels_to_invert (list): 要取反的通道索引，例如 [0] 表示只取反 R 通道。

    返回:
        Tensor: 修改后的图像张量。
    """

    result = image.clone()

    for ch in channels_to_invert:
        if ch < 0 or ch > 2:
            raise ValueError(f"Invalid channel index: {ch}")
        result[..., ch] = 1.0 - result[..., ch]

    return result


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    将 RGB 图像转换为灰度图。

    参数:
        image (Tensor): 形状为 (B, H, W, 3)，值在 [0.0, 1.0] 的 float 类型张量。

    返回:
        Tensor: 形状为 (B, H, W, 1) 的灰度图张量。
    """

    # 定义加权系数
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device).view(1, 1, 1, 3)

    # 加权求和
    grayscale = (image * weights).sum(dim=-1, keepdim=True)  # 结果 shape: (B, H, W, 1)

    return grayscale.expand(-1,-1,-1,3)

class LS_ColorNegative:

    def __init__(self):
        self.NODE_NAME = 'ColorNegative'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "negative_channel" : (negative_channel_list,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_negative'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_negative(self, image, negative_channel,):
        if image.shape[3] == 4:
            rgb_image = image[..., :3]
        else:
            rgb_image = image

        if negative_channel == "RGB":
            ret_image = invert_specific_channel(rgb_image, [0, 1, 2])
        elif negative_channel == "Mono":
            mono_image = rgb_to_grayscale(rgb_image)
            ret_image = invert_specific_channel(mono_image, [0, 1, 2])
        elif negative_channel == "R":
            ret_image = invert_specific_channel(rgb_image, [0])
        elif negative_channel == "G":
            ret_image = invert_specific_channel(rgb_image, [1])
        elif negative_channel == "B":
            ret_image = invert_specific_channel(rgb_image, [2])

        return (ret_image,)


NODE_CLASS_MAPPINGS = {
    "LayerColor: Negative": LS_ColorNegative
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: Negative": "LayerColor: Negative"
}