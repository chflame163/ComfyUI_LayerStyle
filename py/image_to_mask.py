import torch
from PIL import Image, ImageChops
from .imagefunc import log, tensor2pil, image2mask, image_channel_split, normalize_gray, adjust_levels



class ImageToMask:
    def __init__(self):
        self.NODE_NAME = 'ImageToMask'
    @classmethod
    def INPUT_TYPES(s):
        channel_list = ["L(LAB)", "A(Lab)", "B(Lab)",
                        "R(RGB)", "G(RGB)", "B(RGB)", "alpha",
                        "Y(YUV)", "U(YUV)", "V(YUV)",
                        "H(HSV)", "S(HSV", "V(HSV)"]
        return {
            "required": {
                "image": ("IMAGE", ),
                "channel": (channel_list,),
                "black_point": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "white_point": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gray_point": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 9.99, "step": 0.01}),
                "invert_output_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_to_mask"
    CATEGORY = '😺dzNodes/LayerMask'

    def image_to_mask(self, image, channel,
                      black_point, white_point, gray_point,
                      invert_output_mask, mask=None
                      ):

        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_images)):
            orig_image = l_images[i] if i < len(l_images) else l_images[-1]
            orig_image = tensor2pil(orig_image)
            orig_mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            mask = Image.new('L', orig_image.size, 'black')
            if channel == "L(LAB)":
                mask, _, _, _ = image_channel_split(orig_image, 'LAB')
            elif channel == "A(Lab)":
                _, mask, _, _ = image_channel_split(orig_image, 'LAB')
            elif channel == "B(Lab)":
                _, _, mask, _ = image_channel_split(orig_image, 'LAB')
            elif channel == "R(RGB)":
                mask, _, _, _ = image_channel_split(orig_image, 'RGB')
            elif channel == "G(RGB)":
                _, mask, _, _ = image_channel_split(orig_image, 'RGB')
            elif channel == "B(RGB)":
                _, _, mask, _ = image_channel_split(orig_image, 'RGB')
            elif channel == "alpha":
                _, _, _, mask = image_channel_split(orig_image, 'RGBA')
            elif channel == "Y(YUV)":
                mask, _, _, _ = image_channel_split(orig_image, 'YCbCr')
            elif channel == "U(YUV)":
                _, mask, _, _ = image_channel_split(orig_image, 'YCbCr')
            elif channel == "V(YUV)":
                _, _, mask, _ = image_channel_split(orig_image, 'YCbCr')
            elif channel == "H(HSV)":
                mask, _, _, _ = image_channel_split(orig_image, 'HSV')
            elif channel == "S(HSV)":
                _, mask, _, _ = image_channel_split(orig_image, 'HSV')
            elif channel == "V(HSV)":
                _, _, mask, _ = image_channel_split(orig_image, 'HSV')
            mask = normalize_gray(mask)
            mask = adjust_levels(mask, black_point, white_point, gray_point,
                          0, 255)
            if invert_output_mask:
                mask =  ImageChops.invert(mask)
            ret_mask = Image.new('L', mask.size, 'black')
            ret_mask.paste(mask, mask=orig_mask)

            ret_mask = image2mask(ret_mask)

            ret_masks.append(ret_mask)

        return (torch.cat(ret_masks, dim=0), )


NODE_CLASS_MAPPINGS = {
    "LayerMask: ImageToMask": ImageToMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: ImageToMask": "LayerMask: Image To Mask"
}