import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage
from typing import Union, List
from PIL import Image, ImageFilter, ImageChops

def log(message):
    name = 'Layer Style'
    print(f"# 😺dzNodes: {name} ->  {message}")

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image2mask(image:Image) -> torch.Tensor:
    _image = image.convert('RGBA')
    alpha = _image.split() [0]
    bg = Image.new("L", _image.size)
    _image = Image.merge('RGBA', (bg, bg, bg, alpha))
    return pil2tensor(_image)[0, :, :, 3]

def mask2image(mask:torch.Tensor)  -> Image:
    masks = tensor2np(mask)
    # images = []
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color='white')
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image

def shift_image(image:Image, distance_x:int, distance_y:int) -> Image:
    bkcolor = (0, 0, 0)
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=bkcolor)
    for x in range(width):
        for y in range(height):
            if x + distance_x < width and y + distance_y < height:
                pixel = image.getpixel((x + distance_x, y + distance_y))
                ret_image.putpixel((x, y), pixel)
    return ret_image

def chop_image(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:

    ret_image = background_image
    if blend_mode == 'normal':
        ret_image = layer_image
    if blend_mode == 'multply':
        ret_image = ImageChops.multiply(background_image,layer_image)
    if blend_mode == 'screen':
        ret_image = ImageChops.screen(background_image, layer_image)
    if blend_mode == 'add':
        ret_image = ImageChops.add(background_image, layer_image, 1, 0)
    if blend_mode == 'subtract':
        ret_image = ImageChops.subtract(background_image, layer_image, 1, 0)
    if blend_mode == 'difference':
        ret_image = ImageChops.difference(background_image, layer_image)
    if blend_mode == 'darker':
        ret_image = ImageChops.darker(background_image, layer_image)
    if blend_mode == 'lighter':
        ret_image = ImageChops.lighter(background_image, layer_image)

    # opacity
    if opacity == 0:
        ret_image = background_image
    elif opacity < 100:
        alpha = 1.0 - float(opacity) / 100
        ret_image = Image.blend(ret_image, background_image, alpha)

    return ret_image

def expand_mask(mask:torch.Tensor, grow:int, blur:int, expandrate:int) -> torch.Tensor:
    # grow
    c = 0
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in growmask:
        output = m.numpy()
        for _ in range(abs(grow)):
            if grow < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        if grow < 0:
            grow -= abs(expandrate)
        else:
            grow += abs(expandrate)
        output = torch.from_numpy(output)
        out.append(output)
    # blur
    if blur != 0:
        for idx, tensor in enumerate(out):
            pil_image = tensor2pil(tensor.cpu().detach())
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
            out[idx] = pil2tensor(pil_image)

    blurred = torch.cat(out, dim=0)
    return blurred

class DropShadow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        chop_mode = ['normal','multply','screen','add','subtract','difference','darker','lighter']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "layer_mask": ("MASK",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),  # 透明度
                "distance_x": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),  # x_偏移
                "distance_y": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),  # y_偏移
                "grow": ("INT", {"default": 2, "min": -9999, "max": 9999, "step": 1}),  # 扩散
                "blur": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}),  # 模糊
                "shadow_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                # "test_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "shadow_mask",)
    FUNCTION = 'drop_shadow'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def drop_shadow(self, background_image, layer_image, layer_mask,
                  invert_mask, blend_mode, opacity, distance_x, distance_y,
                  grow, blur, shadow_color,
                  ):
        distance_x = -distance_x
        distance_y = -distance_y
        # 处理阴影mask
        if invert_mask:
            layer_mask = 1 - layer_mask
        _layer = tensor2pil(layer_image)
        _mask = mask2image(layer_mask)
        if distance_x != 0 or distance_y != 0:
            _mask = shift_image(_mask, distance_x, distance_y)  # 位移
        shadow_mask = expand_mask(image2mask(_mask), grow, blur, 0)  #扩边，模糊，膨胀

        # 合成阴影
        shadow_color = Image.new("RGB", _layer.size, color=shadow_color)
        alpha = tensor2pil(shadow_mask).convert('L')
        _canvas = tensor2pil(background_image)
        _shadow = chop_image(tensor2pil(background_image), shadow_color, blend_mode, opacity)
        _canvas.paste(_shadow, mask=alpha)

        # 合成layer
        alpha = tensor2pil(layer_mask).convert('L')
        _canvas.paste(_layer, mask=alpha)

        ret_image = _canvas
        ret_mask = shadow_mask
        return (pil2tensor(ret_image), ret_mask,)

NODE_CLASS_MAPPINGS = {
    "LayerStyle_DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle_DropShadow": "Layer Style: Drop Shadow"
}