# code from https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils

import torch
import numpy as np
from PIL import Image
import cv2
from .imagefunc import log, fit_resize_image, tensor2pil, pil2tensor


def resize_img(img, resolution, interpolation=cv2.INTER_CUBIC):
    # print(img)

    # print(resolution)
    return cv2.resize(img, resolution, interpolation=interpolation)

def create_image_from_color(width, height, color=(255, 255, 255)):
    # OpenCV uses BGR, so convert hex color to BGR if necessary
    if isinstance(color, str) and color.startswith('#'):
        color = tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))[::-1]

    # Create a blank image with the specified color
    blank_image = np.full((height, width, 3), color, dtype=np.uint8)
    return blank_image

def fit_image(image, mask=None, output_length=1536, patch_mode="auto"):
    image = image.detach().cpu().numpy()
    if mask is not None:
        mask = mask.detach().cpu().numpy()

    base_length = int(output_length / 3 * 2)
    half_length = int(output_length / 2)
    image_height, image_width, _ = image.shape

    target_width = int(half_length)
    target_height = int(base_length)

    if patch_mode == "auto":
        if image_width > image_height:
            patch_mode = "patch_bottom"
            target_width = int(base_length)
            target_height = int(half_length)
        else:
            patch_mode = "patch_right"
    elif patch_mode == "patch_bottom":
        target_width = int(base_length)
        target_height = int(half_length)

    # 等比例缩放并填充逻辑
    scale_ratio = min(target_width / image_width, target_height / image_height)

    # 计算缩放后的尺寸
    new_width = int(image_width * scale_ratio)
    new_height = int(image_height * scale_ratio)

    # 缩放图片
    image = resize_img(image, (new_width, new_height))

    if mask is not None:
        mask = resize_img(mask, (new_width, new_height), cv2.INTER_NEAREST_EXACT)

    # 计算填充的差值
    diff_x = target_width - new_width
    diff_y = target_height - new_height

    # 计算填充上下左右的像素
    pad_x = diff_x // 2
    pad_y = diff_y // 2

    # 添加白色填充到图片，黑色填充到掩码
    resized_image = cv2.copyMakeBorder(
        image,
        pad_y, diff_y - pad_y,
        pad_x, diff_x - pad_x,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    if mask is not None:
        resized_mask = cv2.copyMakeBorder(
            mask,
            pad_y, diff_y - pad_y,
            pad_x, diff_x - pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    else:
        resized_mask = torch.zeros((target_width, target_height))

    return resized_image, resized_mask, target_width, target_height, patch_mode

def crop_and_scale_as(image:Image, size:tuple):

        target_width, target_height = size
        _image = Image.new('RGB', size=size, color='black')

        ret_image = fit_resize_image(image, target_width, target_height, "crop", Image.LANCZOS)
        return ret_image


class ICMask_Data:
    def __init__(self, x_offset, y_offset, target_width, target_height, total_width, total_height, orig_width, orig_height):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.target_width = target_width
        self.target_height = target_height
        self.total_width = total_width
        self.total_height = total_height
        self.orig_width = orig_width
        self.orig_height = orig_height


class LS_ICMask:
    def __init__(self):
        self.NODE_NAME = 'IC_Mask'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "first_image": ("IMAGE",),
                "patch_mode": (["auto", "patch_right", "patch_bottom"], {
                    "default": "auto",
                }),
                "output_length": ("INT", {
                    "default": 1536,
                }),
                "patch_color": (["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"], {
                    "default": "#FFFFFF",
                }),
            },
            "optional": {
                "first_mask": ("MASK",),
                "second_image": ("IMAGE",),
                "second_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "ICMASK_DATA",)
    RETURN_NAMES = ("image", "mask", "icmask_data",)
    FUNCTION = "ic_mask"
    CATEGORY = '😺dzNodes/LayerUtility'

    def ic_mask(self, first_image, patch_mode, output_length, patch_color, first_mask=None, second_image=None,
                 second_mask=None):
        orig_width = 0
        orig_height = 0
        if output_length % 64 != 0:
            output_length = output_length - (output_length % 64)

        image1 = first_image[0]
        if first_mask is None:
            image1_mask = torch.zeros((image1.shape[0], image1.shape[1]))
        else:
            image1_mask = first_mask[0]

        image1, image1_mask, target_width, target_height, patch_mode = fit_image(image1, image1_mask, output_length,
                                                                                 patch_mode)
        if second_image is not None:
            image2 = second_image[0]
            if second_mask is None:
                image2_mask = torch.zeros((image2.shape[0], image2.shape[1]))
            else:
                image2_mask = second_mask[0]
            orig_width = image2.shape[1]
            orig_height = image2.shape[0]
            image2, image2_mask, _, _, _ = fit_image(image2, image2_mask, output_length, patch_mode)
        else:
            image2 = create_image_from_color(target_width, target_height, color=patch_color)
            image2 = torch.from_numpy(image2)
            if second_mask is None:
                image2_mask = torch.zeros((image2.shape[0], image2.shape[1]))
            else:
                image2_mask = second_mask[0]
            orig_width = image2.shape[1]
            orig_height = image2.shape[0]
            image2, image2_mask, _, _, _ = fit_image(image2, image2_mask, output_length)

        min_y = 0
        min_x = 0

        if second_mask is None or np.all(image2_mask == 0):
            image2_mask = torch.ones((image1.shape[0], image1.shape[1]))

        if patch_mode == "patch_right":
            concatenated_image = np.hstack((image1, image2))
            concatenated_mask = np.hstack((image1_mask, image2_mask))
            min_x = 50
        else:
            concatenated_image = np.vstack((image1, image2))
            concatenated_mask = np.vstack((image1_mask, image2_mask))
            min_y = 50
        min_y = int(min_y / 100.0 * concatenated_image.shape[0])
        min_x = int(min_x / 100.0 * concatenated_image.shape[1])

        return_masks = torch.from_numpy(concatenated_mask)[None,]

        concatenated_image = np.clip(255. * concatenated_image, 0, 255).astype(np.float32) / 255.0
        concatenated_image = torch.from_numpy(concatenated_image)[None,]

        return_images = concatenated_image
        icmask_data = ICMask_Data(min_x, min_y, target_width, target_height, concatenated_image.shape[1],
                                  concatenated_image.shape[0], orig_width, orig_height)

        return (return_images, return_masks, icmask_data)


class LS_ICMask_CropBack:

    def __init__(self):
        self.NODE_NAME = 'IC_Mask_Crop_Back'

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "icmask_data": ("ICMASK_DATA",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_back"
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_back(self, image, icmask_data):
        width = icmask_data.target_width
        height = icmask_data.target_height
        x = icmask_data.x_offset
        y = icmask_data.y_offset
        orig_width = icmask_data.orig_width
        orig_height = icmask_data.orig_height
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:,y:to_y, x:to_x, :]
        pil_image = tensor2pil(img)
        ret_image = crop_and_scale_as(pil_image, (orig_width, orig_height))
        return (pil2tensor(ret_image,),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ICMask": LS_ICMask,
    "LayerUtility: ICMaskCropBack": LS_ICMask_CropBack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ICMask": "LayerUtility: IC Mask",
    "LayerUtility: ICMaskCropBack": "LayerUtility: IC Mask Crop Back",
}