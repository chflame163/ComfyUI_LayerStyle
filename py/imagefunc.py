"""Image process functions for ComfyUI nodes
by chflame https://github.com/chflame163

@author: chflame
@title: LayerStyle
@nickname: LayerStyle
@description: A set of nodes for ComfyUI that can composite layer and mask to achieve Photoshop like functionality.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pickle
import copy
import re
import json
import math
import glob
import numpy as np
import torch
import scipy.ndimage
import cv2
import random
import time
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache
from typing import Union, List
from PIL import Image, ImageFilter, ImageChops, ImageDraw, ImageOps, ImageEnhance, ImageFont
from skimage import img_as_float, img_as_ubyte
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
from colorsys import rgb_to_hsv
import folder_paths
import comfy.model_management
from .blendmodes import *

def log(message:str, message_type:str='info'):
    name = 'LayerStyle'

    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# 😺dzNodes: {name} -> {message}")

try:
    from cv2.ximgproc import guidedFilter
except ImportError as e:
    # print(e)
    log(f"Cannot import name 'guidedFilter' from 'cv2.ximgproc'"
        f"\nA few nodes cannot works properly, while most nodes are not affected. Please REINSTALL package 'opencv-contrib-python'."
        f"\nFor detail refer to \033[4mhttps://github.com/chflame163/ComfyUI_LayerStyle/issues/5\033[0m")



'''warpper'''

# create a wrapper function that can apply a function to multiple images in a batch while passing all other arguments to the function
def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = []
        for img in image:
            images.append(func(self, img, *args, **kwargs))
        batch_tensor = torch.cat(images, dim=0)
        return (batch_tensor,)
    return wrapper


'''pickle'''


def read_image(filename:str) -> Image:
    return Image.open(filename)

def pickle_to_file(obj:object, file_path:str):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file_name:str) -> object:
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_light_leak_images() -> list:
    file = os.path.join(folder_paths.models_dir, "layerstyle", "light_leak.pkl")
    return load_pickle(file)

def check_and_download_model(model_path, repo_id):
    model_path = os.path.join(folder_paths.models_dir, model_path)

    if not os.path.exists(model_path):
        print(f"Downloading {repo_id} model...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt", "onnx", ".git"])
    return model_path

'''Converter'''

def cv22ski(cv2_image:np.ndarray) -> np.array:
    return img_as_float(cv2_image)

def ski2cv2(ski:np.array) -> np.ndarray:
    return img_as_ubyte(ski)

def cv22pil(cv2_img:np.ndarray) -> Image:
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def pil2cv2(pil_img:Image) -> np.array:
    np_img_array = np.asarray(pil_img)
    return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def np2pil(np_image:np.ndarray) -> Image:
    return Image.fromarray(np_image)

def pil2np(pil_image:Image) -> np.array:
    return np.ndarray(pil_image)

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

def tensor2cv2(image:torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)

def image2mask(image:Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def mask2image(mask:torch.Tensor)  -> Image:
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color='white')
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image

'''Image Functions'''

# 颜色加深
def blend_color_burn(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_2) / (img_1 + 0.001)
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 颜色减淡
def blend_color_dodge(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 / (1.0 - img_1 + 0.001)
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 线性加深
def blend_linear_burn(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2 - 1
    mask_1 = img < 0
    img = img * (1 - mask_1)
    return cv22pil(ski2cv2(img))

# 线性减淡
def blend_linear_dodge(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask_2 = img > 1
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

# 变亮
def blend_lighten(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 - img_2
    mask = img > 0
    img = img_1 * mask + img_2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 变暗
def blend_dark(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 - img_2
    mask = img < 0
    img = img_1 * mask + img_2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 滤色
def blend_screen(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = 1 - (1 - img_1) * (1 - img_2)
    return cv22pil(ski2cv2(img))

# 叠加
def blend_overlay(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1 - mask) * (1 - 2 * (1 - img_1) * (1 - img_2))
    return cv22pil(ski2cv2(img))

# 柔光
def blend_soft_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = (2 * img_1 - 1) * (img_2 - img_2 * img_2) + img_2
    T2 = (2 * img_1 - 1) * (np.sqrt(img_2) - img_2) + img_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 强光
def blend_hard_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 2 * img_1 * img_2
    T2 = 1 - 2 * (1 - img_1) * (1 - img_2)
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 亮光
def blend_vivid_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask = img_1 < 0.5
    T1 = 1 - (1 - img_2) / (2 * img_1 + 0.001)
    T2 = img_2 / (2 * (1 - img_1) + 0.001)
    mask_1 = T1 < 0
    mask_2 = T2 > 1
    T1 = T1 * (1 - mask_1)
    T2 = T2 * (1 - mask_2) + mask_2
    img = T1 * mask + T2 * (1 - mask)
    return cv22pil(ski2cv2(img))

# 点光
def blend_pin_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    mask_1 = img_2 < (img_1 * 2 - 1)
    mask_2 = img_2 > 2 * img_1
    T1 = 2 * img_1 - 1
    T2 = img_2
    T3 = 2 * img_1
    img = T1 * mask_1 + T2 * (1 - mask_1) * (1 - mask_2) + T3 * mask_2
    return cv22pil(ski2cv2(img))

# 线性光
def blend_linear_light(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_2 + img_1 * 2 - 1
    mask_1 = img < 0
    mask_2 = img > 1
    img = img * (1 - mask_1)
    img = img * (1 - mask_2) + mask_2
    return cv22pil(ski2cv2(img))

def blend_hard_mix(background_image:Image, layer_image:Image) -> Image:
    img_1 = cv22ski(pil2cv2(background_image))
    img_2 = cv22ski(pil2cv2(layer_image))
    img = img_1 + img_2
    mask = img_1 + img_2 > 1
    img = img * (1 - mask) + mask
    img = img * mask
    return cv22pil(ski2cv2(img))

def shift_image(image:Image, distance_x:int, distance_y:int, background_color:str='#000000', cyclic:bool=False) -> Image:
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=background_color)
    for x in range(width):
        for y in range(height):
            if cyclic:
                    orig_x = x + distance_x
                    if orig_x > width-1 or orig_x < 0:
                        orig_x = abs(orig_x % width)
                    orig_y = y + distance_y
                    if orig_y > height-1 or orig_y < 0:
                        orig_y = abs(orig_y % height)

                    pixel = image.getpixel((orig_x, orig_y))
                    ret_image.putpixel((x, y), pixel)
            else:
                if x > -distance_x and y > -distance_y:  # 防止回转
                    if x + distance_x < width and y + distance_y < height:  # 防止越界
                        pixel = image.getpixel((x + distance_x, y + distance_y))
                        ret_image.putpixel((x, y), pixel)
    return ret_image

def chop_image(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:
    ret_image = background_image
    if blend_mode == 'normal':
        ret_image = copy.deepcopy(layer_image)
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
    if blend_mode == 'color_burn':
        ret_image = blend_color_burn(background_image, layer_image)
    if blend_mode == 'color_dodge':
        ret_image = blend_color_dodge(background_image, layer_image)
    if blend_mode == 'linear_burn':
        ret_image = blend_linear_burn(background_image, layer_image)
    if blend_mode == 'linear_dodge':
        ret_image = blend_linear_dodge(background_image, layer_image)
    if blend_mode == 'overlay':
        ret_image = blend_overlay(background_image, layer_image)
    if blend_mode == 'soft_light':
        ret_image = blend_soft_light(background_image, layer_image)
    if blend_mode == 'hard_light':
        ret_image = blend_hard_light(background_image, layer_image)
    if blend_mode == 'vivid_light':
        ret_image = blend_vivid_light(background_image, layer_image)
    if blend_mode == 'pin_light':
        ret_image = blend_pin_light(background_image, layer_image)
    if blend_mode == 'linear_light':
        ret_image = blend_linear_light(background_image, layer_image)
    if blend_mode == 'hard_mix':
        ret_image = blend_hard_mix(background_image, layer_image)
    # opacity
    if opacity == 0:
        ret_image = background_image
    elif opacity < 100:
        alpha = 1.0 - float(opacity) / 100
        ret_image = Image.blend(ret_image, background_image, alpha)
    return ret_image

def chop_image_v2(background_image:Image, layer_image:Image, blend_mode:str, opacity:int) -> Image:

    backdrop_prepped = np.asarray(background_image.convert('RGBA'), dtype=float)
    source_prepped = np.asarray(layer_image.convert('RGBA'), dtype=float)
    blended_np = BLEND_MODES[blend_mode](backdrop_prepped, source_prepped, opacity / 100)

    return Image.fromarray(np.uint8(blended_np)).convert('RGB')

def remove_background(image:Image, mask:Image, color:str) -> Image:
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=color)
    ret_image.paste(image, mask=mask)
    return ret_image

def sharpen(image:Image) -> Image:
    img = pil2cv2(image)
    Laplace_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]], dtype=np.float32)
    ret_image = cv2.filter2D(img, -1, Laplace_kernel)
    return cv22pil(ret_image)

def gaussian_blur(image:Image, radius:int) -> Image:
    # image = image.convert("RGBA")
    ret_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return ret_image

def motion_blur(image:Image, angle:int, blur:int) -> Image:
    angle += 45
    blur *= 5
    image = np.array(pil2cv2(image))
    M = cv2.getRotationMatrix2D((blur / 2, blur / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(blur))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (blur, blur))
    motion_blur_kernel = motion_blur_kernel / blur
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    ret_image = cv22pil(blurred)
    return ret_image

def __apply_vignette(image, vignette):
    # If image needs to be normalized (0-1 range)
    needs_normalization = image.max() > 1
    if needs_normalization:
        image = image.astype(np.float32) / 255
    final_image = np.clip(image * vignette[..., np.newaxis], 0, 1)
    if needs_normalization:
        final_image = (final_image * 255).astype(np.uint8)
    return final_image
def vignette_image(image:Image, intensity: float, center_x: float, center_y: float) -> Image:
    image = pil2tensor(image)
    _, height, width, _ = image.shape
    # Generate the vignette for each image in the batch
    # Create linear space but centered around the provided center point ratios
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x - (2 * center_x - 1), y - (2 * center_y - 1))
    # Calculate distances to the furthest corner
    distances_to_corners = [
        np.sqrt((0 - center_x) ** 2 + (0 - center_y) ** 2),
        np.sqrt((1 - center_x) ** 2 + (0 - center_y) ** 2),
        np.sqrt((0 - center_x) ** 2 + (1 - center_y) ** 2),
        np.sqrt((1 - center_x) ** 2 + (1 - center_y) ** 2)
    ]
    max_distance_to_corner = np.max(distances_to_corners)
    radius = np.sqrt(X ** 2 + Y ** 2)
    radius = radius / (max_distance_to_corner * np.sqrt(2))  # Normalize radius
    opacity = np.clip(intensity, 0, 1)
    vignette = 1 - radius * opacity
    tensor_image = image.numpy()
    # Apply vignette
    vignette_image = __apply_vignette(tensor_image, vignette)
    return tensor2pil(torch.from_numpy(vignette_image).unsqueeze(0))

def RGB2YCbCr(t):
    YCbCr = t.detach().clone()
    YCbCr[:,:,:,0] = 0.2123 * t[:,:,:,0] + 0.7152 * t[:,:,:,1] + 0.0722 * t[:,:,:,2]
    YCbCr[:,:,:,1] = 0 - 0.1146 * t[:,:,:,0] - 0.3854 * t[:,:,:,1] + 0.5 * t[:,:,:,2]
    YCbCr[:,:,:,2] = 0.5 * t[:,:,:,0] - 0.4542 * t[:,:,:,1] - 0.0458 * t[:,:,:,2]
    return YCbCr

def YCbCr2RGB(t):
    RGB = t.detach().clone()
    RGB[:,:,:,0] = t[:,:,:,0] + 1.5748 * t[:,:,:,2]
    RGB[:,:,:,1] = t[:,:,:,0] - 0.1873 * t[:,:,:,1] - 0.4681 * t[:,:,:,2]
    RGB[:,:,:,2] = t[:,:,:,0] + 1.8556 * t[:,:,:,1]
    return RGB

# gaussian blur a tensor image batch in format [B x H x W x C] on H/W (spatial, per-image, per-channel)
def cv_blur_tensor(images, dx, dy):
    if min(dx, dy) > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx // 20 * 2 + 1, dy // 20 * 2 + 1), 0)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1,1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1,-1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        return torch.from_numpy(np_img)

def image_add_grain(image:Image, scale:float=0.5, strength:float=0.5, saturation:float=0.7, toe:float=0.0, seed:int=0) -> Image:

    image = pil2tensor(image.convert("RGB"))
    t = image.detach().clone()
    torch.manual_seed(seed)
    grain = torch.rand(t.shape[0], int(t.shape[1] // scale), int(t.shape[2] // scale), 3)

    YCbCr = RGB2YCbCr(grain)
    YCbCr[:, :, :, 0] = cv_blur_tensor(YCbCr[:, :, :, 0], 3, 3)
    YCbCr[:, :, :, 1] = cv_blur_tensor(YCbCr[:, :, :, 1], 15, 15)
    YCbCr[:, :, :, 2] = cv_blur_tensor(YCbCr[:, :, :, 2], 11, 11)

    grain = (YCbCr2RGB(YCbCr) - 0.5) * strength
    grain[:, :, :, 0] *= 2
    grain[:, :, :, 2] *= 3
    grain += 1
    grain = grain * saturation + grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3) * (1 - saturation)

    grain = torch.nn.functional.interpolate(grain.movedim(-1, 1), size=(t.shape[1], t.shape[2]),
                                            mode='bilinear').movedim(1, -1)
    t[:, :, :, :3] = torch.clip((1 - (1 - t[:, :, :, :3]) * grain) * (1 - toe) + toe, 0, 1)
    return tensor2pil(t)

def filmgrain_image(image:Image, scale:float, grain_power:float,
                    shadows:float, highs:float, grain_sat:float,
                    sharpen:int=1, grain_type:int=4, src_gamma:float=1.0,
                    gray_scale:bool=False, seed:int=0) -> Image:
    # image = pil2tensor(image)
    # grain_type, 1=fine, 2=fine simple, 3=coarse, 4=coarser
    grain_type_index = 3

    # Apply grain
    from .filmgrainer import filmgrainer as fg
    grain_image = fg.process(image, scale=scale, src_gamma=src_gamma, grain_power=grain_power,
                                      shadows=shadows, highs=highs, grain_type=grain_type_index,
                                      grain_sat=grain_sat, gray_scale=gray_scale, sharpen=sharpen, seed=seed)
    return tensor2pil(torch.from_numpy(grain_image).unsqueeze(0))

def __apply_radialblur(image, blur_strength, radial_mask, focus_spread, steps):
    from .filmgrainer import processing as processing_utils
    needs_normalization = image.max() > 1
    if needs_normalization:
        image = image.astype(np.float32) / 255
    blurred_images = processing_utils.generate_blurred_images(image, blur_strength, steps, focus_spread)
    final_image = processing_utils.apply_blurred_images(image, blurred_images, radial_mask)
    if needs_normalization:
        final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)
    return final_image

def radialblur_image(image:Image, blur_strength:float, center_x:float, center_y:float, focus_spread:float, steps:int=5) -> Image:
    width, height = image.size
    image = pil2tensor(image)
    if image.dim() == 4:
        image = image[0]

    # _, height, width, = image.shape
    # Generate the vignette for each image in the batch
    c_x, c_y = int(width * center_x), int(height * center_y)
    # Calculate distances to all corners from the center
    distances_to_corners = [
        np.sqrt((c_x - 0)**2 + (c_y - 0)**2),
        np.sqrt((c_x - width)**2 + (c_y - 0)**2),
        np.sqrt((c_x - 0)**2 + (c_y - height)**2),
        np.sqrt((c_x - width)**2 + (c_y - height)**2)
    ]
    max_distance_to_corner = max(distances_to_corners)
    # Create and adjust radial mask
    X, Y = np.meshgrid(np.arange(width) - c_x, np.arange(height) - c_y)
    radial_mask = np.sqrt(X**2 + Y**2) / max_distance_to_corner
    tensor_image = image.numpy()
    # Apply blur
    blur_image = __apply_radialblur(tensor_image, blur_strength, radial_mask, focus_spread, steps)
    return tensor2pil(torch.from_numpy(blur_image).unsqueeze(0))

def __apply_depthblur(image, depth_map, blur_strength, focal_depth, focus_spread, steps):
    from .filmgrainer import processing as processing_utils
    # Normalize the input image if needed
    needs_normalization = image.max() > 1
    if needs_normalization:
        image = image.astype(np.float32) / 255
    # Normalize the depth map if needed
    depth_map = depth_map.astype(np.float32) / 255 if depth_map.max() > 1 else depth_map
    # Resize depth map to match the image dimensions
    depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(depth_map_resized.shape) > 2:
        depth_map_resized = cv2.cvtColor(depth_map_resized, cv2.COLOR_BGR2GRAY)
    # Adjust the depth map based on the focal plane
    depth_mask = np.abs(depth_map_resized - focal_depth)
    depth_mask = np.clip(depth_mask / np.max(depth_mask), 0, 1)
    # Generate blurred versions of the image
    blurred_images = processing_utils.generate_blurred_images(image, blur_strength, steps, focus_spread)
    # Use the adjusted depth map as a mask for applying blurred images
    final_image = processing_utils.apply_blurred_images(image, blurred_images, depth_mask)
    # Convert back to original range if the image was normalized
    if needs_normalization:
        final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)
    return final_image

def depthblur_image(image:Image, depth_map:Image, blur_strength:float, focal_depth:float, focus_spread:float, steps:int=5) -> Image:
    width, height = image.size
    image = pil2tensor(image)
    depth_map = pil2tensor(depth_map)
    if image.dim() == 4:
        image = image[0]
    if depth_map.dim() == 4:
        depth_map = depth_map[0]
    tensor_image = image.numpy()
    tensor_image_depth = depth_map.numpy()
    # Apply blur
    blur_image = __apply_depthblur(tensor_image, tensor_image_depth, blur_strength, focal_depth, focus_spread, steps)
    return tensor2pil(torch.from_numpy(blur_image).unsqueeze(0))

def fit_resize_image(image:Image, target_width:int, target_height:int, fit:str, resize_sampler:str, background_color:str = '#000000') -> Image:
    image = image.convert('RGB')
    orig_width, orig_height = image.size
    if image is not None:
        if fit == 'letterbox':
            if orig_width / orig_height > target_width / target_height:  # 更宽，上下留黑
                fit_width = target_width
                fit_height = int(target_width / orig_width * orig_height)
            else:  # 更瘦，左右留黑
                fit_height = target_height
                fit_width = int(target_height / orig_height * orig_width)
            fit_image = image.resize((fit_width, fit_height), resize_sampler)
            ret_image = Image.new('RGB', size=(target_width, target_height), color=background_color)
            ret_image.paste(fit_image, box=((target_width - fit_width)//2, (target_height - fit_height)//2))
        elif fit == 'crop':
            if orig_width / orig_height > target_width / target_height:  # 更宽，裁左右
                fit_width = int(orig_height * target_width / target_height)
                fit_image = image.crop(
                    ((orig_width - fit_width)//2, 0, (orig_width - fit_width)//2 + fit_width, orig_height))
            else:   # 更瘦，裁上下
                fit_height = int(orig_width * target_height / target_width)
                fit_image = image.crop(
                    (0, (orig_height-fit_height)//2, orig_width, (orig_height-fit_height)//2 + fit_height))
            ret_image = fit_image.resize((target_width, target_height), resize_sampler)
        else:
            ret_image = image.resize((target_width, target_height), resize_sampler)
    return  ret_image

def __rotate_expand(image:Image, angle:float, SSAA:int=0, method:str="lanczos") -> Image:
    images = pil2tensor(image)
    expand = "true"
    height, width = images[0, :, :, 0].shape

    def rotate_tensor(tensor):
        resize_sampler = Image.LANCZOS
        rotate_sampler = Image.BICUBIC
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
            rotate_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
            rotate_sampler = Image.BILINEAR
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
            rotate_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
            rotate_sampler = Image.NEAREST
        elif method == "nearest":
            resize_sampler = Image.NEAREST
            rotate_sampler = Image.NEAREST
        img = tensor2pil(tensor)
        if SSAA > 1:
            img_us_scaled = img.resize((width * SSAA, height * SSAA), resize_sampler)
            img_rotated = img_us_scaled.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
            img_down_scaled = img_rotated.resize((img_rotated.width // SSAA, img_rotated.height // SSAA), resize_sampler)
            result = pil2tensor(img_down_scaled)
        else:
            img_rotated = img.rotate(angle, rotate_sampler, expand == "true", fillcolor=(0, 0, 0, 0))
            result = pil2tensor(img_rotated)
        return result

    if angle == 0.0 or angle == 360.0:
        return tensor2pil(images)
    else:
        rotated_tensor = torch.stack([rotate_tensor(images[i]) for i in range(len(images))])
        return tensor2pil(rotated_tensor).convert('RGB')

def image_rotate_extend_with_alpha(image:Image, angle:float, alpha:Image=None, method:str="lanczos", SSAA:int=0) -> tuple:
    _image = __rotate_expand(image.convert('RGB'), angle, SSAA, method)
    if angle is not None:
        _alpha = __rotate_expand(alpha.convert('RGB'), angle, SSAA, method)
        ret_image = RGB2RGBA(_image, _alpha)
    else:
        ret_image = _image
    return (_image, _alpha.convert('L'), ret_image)

def create_box_gradient(start_color_inhex:str, end_color_inhex:str, width:int, height:int, scale:int=50) -> Image:
    # scale is percent of border to center for the rectangle
    if scale > 100:
        scale = 100
    elif scale < 1:
        scale = 1
    start_color = Hex_to_RGB(start_color_inhex)
    end_color = Hex_to_RGB(end_color_inhex)
    ret_image = Image.new("RGB", (width, height), start_color)
    draw = ImageDraw.Draw(ret_image)
    step = int(min(width, height) * scale / 100 / 2)
    if step > 0:
        for i in range(step):
            R = int(start_color[0] * (step - i) / step + end_color[0] * i / step)
            G = int(start_color[1] * (step - i) / step + end_color[1] * i / step)
            B = int(start_color[2] * (step - i) / step + end_color[2] * i / step)
            color = (R, G, B)
            draw.rectangle((i, i, width - i, height - i), fill=color)
    draw.rectangle((step, step, width - step, height - step), fill=end_color)
    return ret_image

def create_gradient(start_color_inhex:str, end_color_inhex:str, width:int, height:int, direction:str='bottom') -> Image:
    # direction = one of top, bottom, left, right
    start_color = Hex_to_RGB(start_color_inhex)
    end_color = Hex_to_RGB(end_color_inhex)
    ret_image = Image.new("RGB", (width, height), start_color)
    draw = ImageDraw.Draw(ret_image)
    if direction == 'bottom':
        for i in range(height):
            R = int(start_color[0] * (height - i) / height + end_color[0] * i / height)
            G = int(start_color[1] * (height - i) / height + end_color[1] * i / height)
            B = int(start_color[2] * (height - i) / height + end_color[2] * i / height)
            color = (R, G, B)
            draw.line((0, i, width, i), fill=color)
    elif direction == 'top':
        for i in range(height):
            R = int(end_color[0] * (height - i) / height + start_color[0] * i / height)
            G = int(end_color[1] * (height - i) / height + start_color[1] * i / height)
            B = int(end_color[2] * (height - i) / height + start_color[2] * i / height)
            color = (R, G, B)
            draw.line((0, i, width, i), fill=color)
    elif direction == 'right':
        for i in range(width):
            R = int(start_color[0] * (width - i) / width + end_color[0] * i / width)
            G = int(start_color[1] * (width - i) / width + end_color[1] * i / width)
            B = int(start_color[2] * (width - i) / width + end_color[2] * i / width)
            color = (R, G, B)
            draw.line((i, 0, i, height), fill=color)
    elif direction == 'left':
        for i in range(width):
            R = int(end_color[0] * (width - i) / width + start_color[0] * i / width)
            G = int(end_color[1] * (width - i) / width + start_color[1] * i / width)
            B = int(end_color[2] * (width - i) / width + start_color[2] * i / width)
            color = (R, G, B)
            draw.line((i, 0, i, height), fill=color)
    else:
        log(f'A argument error of imagefunc.create_gradient(), '
            f'"direction=" must one of "top, bottom, left, right".',
            message_type='error')

    return ret_image

def gradient(start_color_inhex:str, end_color_inhex:str, width:int, height:int, angle:float, ) -> Image:
    radius = int((width + height) / 4)
    g = create_gradient(start_color_inhex, end_color_inhex, radius, radius)
    _canvas = Image.new('RGB', size=(radius, radius*3), color=start_color_inhex)
    top = Image.new('RGB', size=(radius, radius), color=start_color_inhex)
    bottom = Image.new('RGB', size=(radius, radius),color=end_color_inhex)
    _canvas.paste(top, box=(0, 0, radius, radius))
    _canvas.paste(g, box=(0, radius, radius, radius * 2))
    _canvas.paste(bottom,box=(0, radius * 2, radius, radius * 3))
    _canvas = _canvas.resize((radius * 3, radius * 3))
    _canvas = __rotate_expand(_canvas,angle)
    center = int(_canvas.width / 2)
    _x = int(width / 3)
    _y = int(height / 3)
    ret_image = _canvas.crop((center - _x, center - _y, center + _x, center + _y))
    ret_image = ret_image.resize((width, height))
    return ret_image

def draw_rounded_rectangle(image:Image, radius:int, bboxes:list, scale_factor:int=2, color:str="white") -> Image:
        """
        绘制圆角矩形图像。
        image:输入图片
        radius: 半径，100为纯椭圆
        bboxes: (x1,y1,x2,y2)列表
        scale_factor: 放大倍数
        :return: 绘制好的pillow图像
        """
        if scale_factor < 1 : scale_factor = 1

        img = image.resize((image.width * scale_factor, image.height * scale_factor), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        for (x1, y1, x2, y2) in bboxes:
            r = radius * min(x2-x1, y2-y1) * 0.005
            x1, y1, x2, y2 = x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor
            # 计算圆角矩形的四个角的圆弧
            draw.rounded_rectangle([x1, y1, x2, y2], radius=r * scale_factor, fill=color)

        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = img.resize((image.width, image.height), Image.LANCZOS)

        return img

def draw_rect(image:Image, x:int, y:int, width:int, height:int, line_color:str, line_width:int,
              box_color:str=None) -> Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + width, y + height), fill=box_color, outline=line_color, width=line_width, )
    return image

def draw_border(image:Image, border_width:int, color:str='#FFFFFF') -> Image:
    return ImageOps.expand(image, border=border_width, fill=color)

# 对灰度图像进行直方图均衡化
def normalize_gray(image:Image) -> Image:
    if image.mode != 'L':
        image = image.convert('L')
    img = np.asarray(image)
    balanced_img = img.copy()
    hist, bins = np.histogram(img.reshape(-1), 256, (0, 256))
    bmin = np.min(np.where(hist > (hist.sum() * 0.0005)))
    bmax = np.max(np.where(hist > (hist.sum() * 0.0005)))
    balanced_img = np.clip(img, bmin, bmax)
    balanced_img = ((balanced_img - bmin) / (bmax - bmin) * 255)
    return Image.fromarray(balanced_img).convert('L')

def remap_pixel(pixel:int, min_brightness:int, max_brightness:int) -> int:
    return int((pixel - min_brightness) / (max_brightness - min_brightness) * 255)
def histogram_range(image:Image, black_point:int, black_range:int, white_point:int, white_range:int) -> Image:

    if image.mode != 'L':
        image = image.convert('L')

    if black_point == 255:
        black_point = 254
    if white_point == 0:
        white_point = 1
    if black_point + black_range > 255:
        black_range = 255 - black_point
    if white_range > white_point:
        white_range = white_point

    white_image = Image.new("L", size=image.size, color="white")
    black_image = Image.new("L", size=image.size, color="black")

    if black_point == white_point:
        return white_image


    # draw white part
    white_part = black_image
    if white_point < 255 or white_range > 0:
        for y in (range(image.height)):
            for x in range(image.width):
                pixel = image.getpixel((x, y))
                if pixel > white_point: # put white
                    white_part.putpixel((x, y), 255)
                elif pixel > white_point - white_range:
                    pixel = remap_pixel(pixel, white_point - white_range, white_point)
                    white_part.putpixel((x, y), pixel)
    white_part = ImageChops.invert(white_part)


    # draw black part
    black_part = black_image
    if black_point > 0 or black_range > 0:
        for y in (range(image.height)):
            for x in range(image.width):
                pixel = image.getpixel((x, y))
                if pixel < black_point: # put black
                    black_part.putpixel((x, y), 255)
                elif pixel < black_point + black_range:
                    pixel = remap_pixel(pixel, black_point, black_point + black_range)
                    black_part.putpixel((x, y), 255 - pixel)
    black_part = ImageChops.invert(black_part)

    ret_image = chop_image_v2(white_part, black_part, blend_mode='darken', opacity=100)

    return ret_image

def histogram_equalization(image:Image, mask:Image=None, gamma_strength=0.5) -> Image:

    if image.mode != 'L':
        image = image.convert('L')

    if mask is not None:
        if mask.mode != 'L':
            mask = mask.convert('L')
    else:
        mask = Image.new('L', size=image.size, color = 'white')

    # calculate Min/Max brightness pixel
    min_brightness = 255
    max_brightness = 0
    average_brightness = 0
    total_pixel = 0
    for y in range(image.height):
        for x in range(image.width):
            if mask.getpixel((x, y)) == 0:
                continue
            else:
                pixel = image.getpixel((x, y))
                if pixel < min_brightness:
                    min_brightness = pixel
                if pixel > max_brightness:
                    max_brightness = pixel
                average_brightness += pixel
                total_pixel += 1
    if total_pixel == 0:
        log(f"histogram_equalization: mask is not available, return orinianl image.")
        return image
    average_brightness = int(average_brightness / total_pixel)

    for y in range(image.height):
        for x in range(image.width):
            pixel = image.getpixel((x, y))
            image.putpixel((x, y), remap_pixel(pixel, min_brightness, max_brightness))

    image = gamma_trans(image, (average_brightness - 127) / 127 * gamma_strength * 0.66 + 1)

    return image.convert('L')

def adjust_levels(image:Image, input_black:int=0, input_white:int=255, midtones:float=1.0,
                  output_black:int=0, output_white:int=255) -> Image:

    if input_black == input_white or output_black == output_white:
        return Image.new('RGB', size=image.size, color='gray')

    img = pil2cv2(image).astype(np.float64)

    if input_black > input_white:
        input_black, input_white = input_white, input_black
    if output_black > output_white:
        output_black, output_white = output_white, output_black


    # input_levels remap
    if input_black > 0 or input_white < 255:
        img = 255 * ((img - input_black) / (input_white - input_black))
        img[img < 0] = 0
        img[img > 255] = 255

    # # mid_tone
    if midtones != 1.0:
        img = 255 * np.power(img / 255, 1.0 / midtones)

        img[img < 0] = 0
        img[img > 255] = 255

    # output_levels remap
    if output_black > 0 or output_white < 255:
        img = (img / 255) * (output_white - output_black) + output_black
        img[img < 0] = 0
        img[img > 255] = 255

    img = img.astype(np.uint8)
    return cv22pil(img)

def get_image_color_tone(image:Image, mask:Image=None) -> str:
    image = image.convert('RGB')
    max_score = 0.0001
    dominant_color = (255, 255, 255)
    if mask is not None:
        if mask.mode != 'L':
            mask = mask.convert('L')
        canvas = Image.new('RGB', size=image.size, color='black')
        canvas.paste(image, mask=mask)
        image = canvas

    all_colors = image.getcolors(image.width * image.height)
    for count, (r, g, b) in all_colors:
        if mask is not None:
            if r + g + b < 2:  # 忽略黑色
                continue
        saturation = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13,235)
        y = (y - 16.0) / (235 - 16)
        score = (saturation+0.1)*count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    ret_color = RGB_to_Hex(dominant_color)
    return ret_color

def get_image_color_average(image:Image, mask:Image=None) -> str:
    image = image.convert('RGB')
    width, height = image.size
    total_red = 0
    total_green = 0
    total_blue = 0
    total_pixel =0
    for y in range(height):
        for x in range(width):
            if mask is not None:
                if mask.mode != 'L':
                    mask = mask.convert('L')
                if mask.getpixel((x, y)) <= 127:
                    continue
            rgb = image.getpixel((x, y))
            total_red += rgb[0]
            total_green += rgb[1]
            total_blue += rgb[2]
            total_pixel += 1

    average_red = total_red // total_pixel
    average_green = total_green // total_pixel
    average_blue = total_blue // total_pixel
    color = (average_red, average_green, average_blue)
    ret_color = RGB_to_Hex(color)
    return ret_color

def get_gray_average(image:Image, mask:Image=None) -> int:
    # image.mode = 'HSV', mask.mode = 'L'
    image = image.convert('HSV')

    if mask is not None:
        if mask.mode != 'L':
            mask = mask.convert('L')
    else:
        mask = Image.new('L', size=image.size, color='white')
    _, _, _v = image.convert('HSV').split()
    _v = np.array(_v)
    average_gray = _v[np.array(mask) > 16].mean()
    # width, height = image.size
    # total_gray = 0
    # valid_pixels = 0
    # for y in range(height):
    #     for x in range(width):
    #         if mask is not None:
    #             if mask.getpixel((x, y)) > 16:  #mask亮度低于16的忽略不计
    #                 gray = _v.getpixel((x, y))
    #                 total_gray += gray
    #                 valid_pixels += 1
    #         else:
    #             gray = _v.getpixel((x, y))
    #             total_gray += gray
    #             valid_pixels += 1
    # average_gray = total_gray // valid_pixels
    return average_gray

def calculate_shadow_highlight_level(gray:int) -> float:
    range = 255
    shadow_exponent = 3
    highlight_exponent = 2
    shadow_ratio = gray ** shadow_exponent / range ** shadow_exponent
    highlight_ratio = gray ** highlight_exponent / range ** highlight_exponent
    shadow_level = shadow_ratio * 100 + (1 - shadow_ratio) * 32
    highlight_level = highlight_ratio * 100 + (1 - highlight_ratio) * 32
    return shadow_level, highlight_level

def luminance_keyer(image:Image, low:float=0, high:float=1, gamma:float=1) -> Image:
    image = pil2tensor(image)
    t = image[:, :, :, :3].detach().clone()
    alpha = 0.2126 * t[:, :, :, 0] + 0.7152 * t[:, :, :, 1] + 0.0722 * t[:, :, :, 2]
    if low == high:
        alpha = (alpha > high).to(t.dtype)
    else:
        alpha = (alpha - low) / (high - low)
    if gamma != 1.0:
        alpha = torch.pow(alpha, 1 / gamma)
    alpha = torch.clamp(alpha, min=0, max=1).unsqueeze(3).repeat(1, 1, 1, 3)
    return tensor2pil(alpha).convert('L')

def get_image_bright_average(image:Image) -> int:
    image = image.convert('L')
    width, height = image.size
    total_bright = 0
    pixels = 0
    for y in range(height):
        for x in range(width):
            b = image.getpixel((x, y))
            if b > 1:  # 排除死黑
                pixels += 1
                total_bright += b
    return int(total_bright / pixels)

def image_channel_split(image:Image, mode = 'RGBA') -> tuple:
    _image = image.convert('RGBA')
    channel1 = Image.new('L', size=_image.size, color='black')
    channel2 = Image.new('L', size=_image.size, color='black')
    channel3 = Image.new('L', size=_image.size, color='black')
    channel4 = Image.new('L', size=_image.size, color='black')
    if mode == 'RGBA':
        channel1, channel2, channel3, channel4 = _image.split()
    if mode == 'RGB':
        channel1, channel2, channel3 = _image.convert('RGB').split()
    if mode == 'YCbCr':
        channel1, channel2, channel3 = _image.convert('YCbCr').split()
    if mode == 'LAB':
        channel1, channel2, channel3 = _image.convert('LAB').split()
    if mode == 'HSV':
        channel1, channel2, channel3 = _image.convert('HSV').split()
    return channel1, channel2, channel3, channel4

def image_channel_merge(channels:tuple, mode = 'RGB' ) -> Image:
    channel1 = channels[0].convert('L')
    channel2 = channels[1].convert('L')
    channel3 = channels[2].convert('L')
    channel4 = Image.new('L', size=channel1.size, color='white')
    if mode == 'RGBA':
        if len(channels) > 3:
            channel4 = channels[3].convert('L')
        ret_image = Image.merge('RGBA',[channel1, channel2, channel3, channel4])
    elif mode == 'RGB':
        ret_image = Image.merge('RGB', [channel1, channel2, channel3])
    elif mode == 'YCbCr':
        ret_image = Image.merge('YCbCr', [channel1, channel2, channel3]).convert('RGB')
    elif mode == 'LAB':
        ret_image = Image.merge('LAB', [channel1, channel2, channel3]).convert('RGB')
    elif mode == 'HSV':
        ret_image = Image.merge('HSV', [channel1, channel2, channel3]).convert('RGB')
    return ret_image

def image_gray_offset(image:Image, offset:int) -> Image:
    image = image.convert('L')
    image_array = np.array(image, dtype=np.int16)
    image_array = np.clip(image_array + offset, 0, 255).astype(np.uint8)
    ret_image = Image.fromarray(image_array, mode='L')
    return ret_image

def image_gray_ratio(image:Image, ratio:float) -> Image:
    image = image.convert('L')
    image_array = np.array(image, dtype=np.float32)
    image_array = np.clip(image_array * ratio, 0, 255).astype(np.uint8)
    ret_image = Image.fromarray(image_array, mode='L')
    return ret_image

def image_hue_offset(image:Image, offset:int) -> Image:
    image = image.convert('L')
    image_array = np.array(image, dtype=np.int16)
    image_array = (image_array + offset) % 256
    image_array = image_array.astype(np.uint8)
    ret_image = Image.fromarray(image_array, mode='L')

    return ret_image

def gamma_trans(image:Image, gamma:float) -> Image:
    cv2_image = pil2cv2(image)
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    _corrected = cv2.LUT(cv2_image,gamma_table)
    return cv22pil(_corrected)


def read_LUT_IridasCube_encode_utf8(path: str):
    from colour.utilities import as_float_array, as_int_scalar
    from colour.io.luts.lut import LUT3x1D, LUT3D
    title = re.sub("_|-|\\.", " ", os.path.splitext(os.path.basename(path))[0])
    domain_min, domain_max = np.array([0, 0, 0]), np.array([1, 1, 1])
    dimensions: int = 3
    size: int = 2
    data = []
    comments = []

    with open(path, encoding='utf-8') as cube_file:
        lines = cube_file.readlines()
        for line in lines:

            line = line.strip()  # noqa: PLW2901

            if len(line) == 0:
                continue

            if line.startswith("#"):
                comments.append(line[1:].strip())
                continue

            tokens = line.split()
            if tokens[0] == "TITLE":
                title = " ".join(tokens[1:])[1:-1]
            elif tokens[0] == "DOMAIN_MIN":
                domain_min = as_float_array(tokens[1:])
            elif tokens[0] == "DOMAIN_MAX":
                domain_max = as_float_array(tokens[1:])
            elif tokens[0] == "LUT_1D_SIZE":
                dimensions = 2
                size = as_int_scalar(tokens[1])
            elif tokens[0] == "LUT_3D_SIZE":
                dimensions = 3
                size = as_int_scalar(tokens[1])
            else:
                data.append(tokens)

    table = as_float_array(data)

    LUT: LUT3x1D | LUT3D
    if dimensions == 2:
        LUT = LUT3x1D(
            table,
            title,
            np.vstack([domain_min, domain_max]),
            comments=comments,
        )
    elif dimensions == 3:
        # The lines of table data shall be in ascending index order,
        # with the first component index (Red) changing most rapidly,
        # and the last component index (Blue) changing least rapidly.
        table = table.reshape([size, size, size, 3], order="F")

        LUT = LUT3D(
            table,
            title,
            np.vstack([domain_min, domain_max]),
            comments=comments,
        )

    return LUT


def apply_lut(image:Image, lut_file:str, colorspace:str, strength:int, clip_values:bool=True) -> Image:
    """
    Apply a LUT to an image.
    :param image: Image to apply the LUT to.
    :param lut_file: LUT file to apply.
    :param colorspace: Colorspace to convert the image to before applying the LUT.
    :param clip_values: Clip the values of the LUT to the domain of the LUT.
    :param strength: Strength of the LUT.
    :return: Image with the LUT applied.
    """
    log_colorspace = False
    if colorspace == "log":
        log_colorspace = True

    # from colour.io.luts.iridas_cube import read_LUT_IridasCube

    lut = read_LUT_IridasCube_encode_utf8(lut_file)
    lut.name = lut_file

    if clip_values:
        if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
            lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
        else:
            if len(lut.table.shape) == 2:  # 3x1D
                for dim in range(3):
                    lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
            else:  # 3D
                for dim in range(3):
                    lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

    img = pil2tensor(image)
    lut_img = img.numpy().copy()
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
    dom_scale = None
    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        lut_img = lut_img * dom_scale + lut.domain[0]
    if log_colorspace:
        lut_img = lut_img ** (1/2.2)
    lut_img = lut.apply(lut_img)
    if log_colorspace:
        lut_img = lut_img ** (2.2)
    if is_non_default_domain:
        lut_img = (lut_img - lut.domain[0]) / dom_scale
    lut_img = torch.from_numpy(lut_img)
    if strength < 100:
        strength /= 100
        lut_img = strength * lut_img + (1 - strength) * img

    return tensor2pil(lut_img)

def color_adapter(image:Image, ref_image:Image) -> Image:
    image = pil2cv2(image)
    ref_image = pil2cv2(ref_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_mean, image_std = calculate_mean_std(image)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB)
    ref_image_mean, ref_image_std = calculate_mean_std(ref_image)
    _image = ((image - image_mean) * (ref_image_std / image_std)) + ref_image_mean
    np.putmask(_image, _image > 255, values=255)
    np.putmask(_image, _image < 0, values=0)
    ret_image = cv2.cvtColor(cv2.convertScaleAbs(_image), cv2.COLOR_LAB2BGR)
    return cv22pil(ret_image)

def calculate_mean_std(image:Image):
    mean, std = cv2.meanStdDev(image)
    mean = np.hstack(np.around(mean, decimals=2))
    std = np.hstack(np.around(std, decimals=2))
    return mean, std

def image_watercolor(image:Image, level:int=50) -> Image:
    img = pil2cv2(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    factor = (level / 128.0) ** 2
    sigmaS= int((image.width + image.height) / 5.0 * factor) + 1
    sigmaR = sigmaS / 32.0 * factor + 0.002
    img_color = cv2.stylization(img, sigma_s=sigmaS, sigma_r=sigmaR)
    ret_image = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return cv22pil(ret_image)


def image_beauty(image:Image, level:int=50) -> Image:
    img = pil2cv2(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    factor = (level / 50.0)**2
    d = int((image.width + image.height) / 256 * factor)
    sigmaColor = max(1, float((image.width + image.height) / 256 * factor))
    sigmaSpace = max(1, float((image.width + image.height) / 160 * factor))
    img_bit = cv2.bilateralFilter(src=img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    ret_image = cv2.cvtColor(img_bit, cv2.COLOR_BGR2RGB)
    return cv22pil(ret_image)


def pixel_spread(image:Image, mask:Image) -> Image:
    from pymatting import estimate_foreground_ml
    i1 = pil2tensor(image)
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')
    i_dup = copy.deepcopy(i1.cpu().numpy().astype(np.float64))
    a_dup = copy.deepcopy(pil2tensor(mask).cpu().numpy().astype(np.float64))
    fg = copy.deepcopy(i1.cpu().numpy().astype(np.float64))

    for index, img in enumerate(i_dup):
        alpha = a_dup[index][:, :, 0]
        fg[index], _ = estimate_foreground_ml(img, np.array(alpha), return_background=True)

    return tensor2pil(torch.from_numpy(fg.astype(np.float32)))


def watermark_image_size(image:Image) -> int:
    size = int(math.sqrt(image.width * image.height * 0.015625) * 0.9)
    return size

def add_invisibal_watermark(image:Image, watermark_image:Image) -> Image:
    """
    Adds an invisible watermark to an image.
    """
    orig_image_mode = image.mode
    temp_dir = os.path.join(folder_paths.get_temp_directory(), generate_random_name('_watermark_', '_temp', 16))
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    image_dir = os.path.join(temp_dir, 'image')
    wm_dir = os.path.join(temp_dir, 'wm')
    result_dir = os.path.join(temp_dir, 'result')

    try:
        os.makedirs(image_dir)
        os.makedirs(wm_dir)
        os.makedirs(result_dir)
    except Exception as e:
        # print(e)
        log(f"Error: {NODE_NAME} skipped, because unable to create temporary folder.", message_type='error')
        return (image,)

    image_file_name = os.path.join(generate_random_name('watermark_orig_', '_temp', 16) + '.png')
    wm_file_name = os.path.join(generate_random_name('watermark_image_', '_temp', 16) + '.png')
    output_file_name = os.path.join(generate_random_name('watermark_output_', '_temp', 16) + '.png')

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(os.path.join(image_dir, image_file_name))
        watermark_image.save(os.path.join(wm_dir, wm_file_name))
    except IOError as e:
        # print(e)
        log(f"Error: {NODE_NAME} skipped, because unable to create temporary file.", message_type='error')
        return (image,)

    from blind_watermark import WaterMark
    bwm1 = WaterMark(password_img=1, password_wm=1)
    bwm1.read_img(os.path.join(image_dir, image_file_name))
    bwm1.read_wm(os.path.join(wm_dir, wm_file_name))
    output_image = os.path.join(result_dir, output_file_name)
    bwm1.embed(output_image, compression_ratio=100)

    return Image.open(output_image).convert(orig_image_mode)

def decode_watermark(image:Image, watermark_image_size:int=94) -> Image:
    temp_dir = os.path.join(folder_paths.get_temp_directory(), generate_random_name('_watermark_', '_temp', 16))
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    image_dir = os.path.join(temp_dir, 'decode_image')
    result_dir = os.path.join(temp_dir, 'decode_result')

    try:
        os.makedirs(image_dir)
        os.makedirs(result_dir)
    except Exception as e:
        # print(e)
        log(f"Error: {NODE_NAME} skipped, because unable to create temporary folder.", message_type='error')
        return (image,)

    image_file_name = os.path.join(generate_random_name('watermark_decode_', '_temp', 16) + '.png')
    output_file_name = os.path.join(generate_random_name('watermark_decode_output_', '_temp', 16) + '.png')

    try:
        image.save(os.path.join(image_dir, image_file_name))
    except IOError as e:
        # print(e)
        log(f"Error: {NODE_NAME} skipped, because unable to create temporary file.", message_type='error')
        return (image,)

    from blind_watermark import WaterMark
    bwm1 = WaterMark(password_img=1, password_wm=1)
    decode_image = os.path.join(image_dir, image_file_name)
    output_image = os.path.join(result_dir, output_file_name)

    try:
        bwm1.extract(filename=decode_image, wm_shape=(watermark_image_size, watermark_image_size),
                     out_wm_name=os.path.join(output_image),)
        ret_image = Image.open(output_image)
    except Exception as e:
        log(f"blind watermark extract fail, {e}")
        ret_image = Image.new("RGB", (64, 64), color="black")
    ret_image = normalize_gray(ret_image)
    return ret_image

# def generate_text_image(text:str, font_path:str, font_size:int, text_color:str="#FFFFFF",
#                         vertical:bool=True, stroke_width:int=1, stroke_color:str="#000000",
#                          spacing:int=0, leading:int=0) -> tuple:
#
#     lines = text.split("\n")
#     if vertical:
#         layout = "vertical"
#     else:
#         layout = "horizontal"
#     char_coordinates = []
#     if layout == "vertical":
#         x = 0
#         y = 0
#         for i in range(len(lines)):
#             line = lines[i]
#             for char in line:
#                 char_coordinates.append((x, y))
#                 y += font_size + spacing
#             x += font_size + leading
#             y = 0
#     else:
#         x = 0
#         y = 0
#         for line in lines:
#             for char in line:
#                 char_coordinates.append((x, y))
#                 x += font_size + spacing
#             y += font_size + leading
#             x = 0
#     if layout == "vertical":
#         width = (len(lines) * (font_size + spacing)) - spacing
#         height = ((len(max(lines, key=len)) + 1) * (font_size + spacing)) + spacing
#     else:
#         width = (len(max(lines, key=len)) * (font_size + spacing)) - spacing
#         height = ((len(lines) - 1) * (font_size + spacing)) + font_size
#
#     image = Image.new('RGBA', size=(width, height), color=stroke_color)
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.truetype(font_path, font_size)
#     index = 0
#     for i, line in enumerate(lines):
#         for j, char in enumerate(line):
#             x, y = char_coordinates[index]
#             if stroke_width > 0:
#                 draw.text((x - stroke_width, y), char, font=font, fill=stroke_color)
#                 draw.text((x + stroke_width, y), char, font=font, fill=stroke_color)
#                 draw.text((x, y - stroke_width), char, font=font, fill=stroke_color)
#                 draw.text((x, y + stroke_width), char, font=font, fill=stroke_color)
#             draw.text((x, y), char, font=font, fill=text_color)
#             index += 1
#     return (image.convert('RGB'), image.split()[3])

def generate_text_image(width:int, height:int, text:str, font_file:str, text_scale:float=1, font_color:str="#FFFFFF",) -> Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_size = int(width / len(text) * text_scale)
    font = ImageFont.truetype(font_file, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = int((width - text_width) / 2)
    y = int((height - text_height) / 2) - int(font_size / 2)
    draw.text((x, y), text, font=font, fill=font_color)
    return image

'''Mask Functions'''

def create_mask_from_color_cv2(image:Image, color:str, tolerance:int=0) -> Image:
    (r, g, b) = Hex_to_RGB(color)
    target_color = (b, g, r)
    tolerance = 127 + int(tolerance * 1.28)
    # tolerance = 255 - tolerance
    # 将RGB颜色转换为HSV颜色空间
    image = pil2cv2(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义目标颜色的HSV范围
    lower_color = np.array([max(target_color[0] - tolerance, 0), max(target_color[1] - tolerance, 0), max(target_color[2] - tolerance, 0)])
    upper_color = np.array([min(target_color[0] + tolerance, 255), min(target_color[1] + tolerance, 255), min(target_color[2] + tolerance, 255)])

    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    return cv22pil(mask).convert("L")

def create_mask_from_color_tensor(image:Image, color:str, tolerance:int=0) -> Image:
    threshold = int(tolerance * 1.28)
    (red, green, blue) = Hex_to_RGB(color)
    image = pil2tensor(image).squeeze()
    temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
    color_value = torch.tensor([red, green, blue])
    lower_bound = (color_value - threshold).clamp(min=0)
    upper_bound = (color_value + threshold).clamp(max=255)
    lower_bound = lower_bound.view(1, 1, 1, 3)
    upper_bound = upper_bound.view(1, 1, 1, 3)
    mask = (temp >= lower_bound) & (temp <= upper_bound)
    mask = mask.all(dim=-1)
    mask = mask.float()
    return tensor2pil(mask).convert("L")

@lru_cache(maxsize=1, typed=False)
def load_RMBG_model():
    from .briarmbg import BriaRMBG
    current_directory = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BriaRMBG()
    model_path = ""
    try:
        model_path = os.path.join(os.path.normpath(folder_paths.folder_names_and_paths['rmbg'][0][0]), "model.pth")
    except:
        pass
    if not os.path.exists(model_path):
        model_path = os.path.join(folder_paths.models_dir, "rmbg", "RMBG-1.4", "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(current_directory), "RMBG-1.4", "model.pth")
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    return net


def RMBG(image:Image) -> Image:
    rmbgmodel = load_RMBG_model()
    w, h = image.size
    im_np = np.array(image.resize((1024, 1024), Image.BILINEAR))
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.divide(torch.unsqueeze(im_tensor, 0), 255.0)
    im_tensor = TF.normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()
    result = rmbgmodel(im_tensor)
    result = torch.squeeze(F.interpolate(result[0][0], size=(h, w), mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
    _mask = torch.from_numpy(np.squeeze(im_array).astype(np.float32))
    return tensor2pil(_mask)

def guided_filter_alpha(image:torch.Tensor, mask:torch.Tensor, filter_radius:int) -> torch.Tensor:
    sigma = 0.15
    d = filter_radius + 1
    mask = pil2tensor(tensor2pil(mask).convert('RGB'))
    if not bool(d % 2):
        d += 1
    s = sigma / 10
    i_dup = copy.deepcopy(image.cpu().numpy())
    a_dup = copy.deepcopy(mask.cpu().numpy())
    for index, image in enumerate(i_dup):
        alpha_work = a_dup[index]
        i_dup[index] = guidedFilter(image, alpha_work, d, s)
    return torch.from_numpy(i_dup)

#pymatting edge detail
def mask_edge_detail(image:torch.Tensor, mask:torch.Tensor, detail_range:int=8, black_point:float=0.01, white_point:float=0.99) -> torch.Tensor:
    from pymatting import fix_trimap, estimate_alpha_cf
    d = detail_range * 5 + 1
    mask = pil2tensor(tensor2pil(mask).convert('RGB'))
    if not bool(d % 2):
        d += 1
    i_dup = copy.deepcopy(image.cpu().numpy().astype(np.float64))
    a_dup = copy.deepcopy(mask.cpu().numpy().astype(np.float64))
    for index, img in enumerate(i_dup):
        trimap = a_dup[index][:, :, 0]  # convert to single channel
        if detail_range > 0:
            trimap = cv2.GaussianBlur(trimap, (d, d), 0)
        trimap = fix_trimap(trimap, black_point, white_point)
        alpha = estimate_alpha_cf(img, trimap, laplacian_kwargs={"epsilon": 1e-6},
                                  cg_kwargs={"maxiter": 500})
        a_dup[index] = np.stack([alpha, alpha, alpha], axis=-1)  # convert back to rgb
    return torch.from_numpy(a_dup.astype(np.float32))

class VITMatteModel:
    def __init__(self,model,processor):
        self.model = model
        self.processor = processor

def load_VITMatte_model(model_name:str, local_files_only:bool=False) -> object:
    model_name = "vitmatte"
    model_repo = "hustvl/vitmatte-small-composition-1k"
    model_path  = check_and_download_model(model_name, model_repo)
    from transformers import VitMatteImageProcessor, VitMatteForImageMatting
    model = VitMatteForImageMatting.from_pretrained(model_path, local_files_only=local_files_only)
    processor = VitMatteImageProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    vitmatte = VITMatteModel(model, processor)
    return vitmatte

def generate_VITMatte(image:Image, trimap:Image, local_files_only:bool=False, device:str="cpu", max_megapixels:float=2.0) -> Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if trimap.mode != 'L':
        trimap = trimap.convert('L')
    max_megapixels *= 1048576
    width, height = image.size
    ratio = width / height
    target_width = math.sqrt(ratio * max_megapixels)
    target_height = target_width / ratio
    target_width = int(target_width)
    target_height = int(target_height)
    if width * height > max_megapixels:
        image = image.resize((target_width, target_height), Image.BILINEAR)
        trimap = trimap.resize((target_width, target_height), Image.BILINEAR)
        # log(f"vitmatte image size {width}x{height} too large, resize to {target_width}x{target_height} for processing.")
    model_name = "hustvl/vitmatte-small-composition-1k"
    if device=="cpu":
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            log("vitmatte device is set to cuda, but not available, using cpu instead.")
            device = torch.device('cpu')
    vit_matte_model = load_VITMatte_model(model_name=model_name, local_files_only=local_files_only)
    vit_matte_model.model.to(device)
    # log(f"vitmatte processing, image size = {image.width}x{image.height}, device = {device}.")
    inputs = vit_matte_model.processor(images=image, trimaps=trimap, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        predictions = vit_matte_model.model(**inputs).alphas
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    mask = tensor2pil(predictions).convert('L')
    mask = mask.crop(
        (0, 0, image.width, image.height))  # remove padding that the prediction appends (works in 32px tiles)
    if width * height > max_megapixels:
        mask = mask.resize((width, height), Image.BILINEAR)
    return mask

def generate_VITMatte_trimap(mask:torch.Tensor, erode_kernel_size:int, dilate_kernel_size:int) -> Image:
    def g_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        eroded = cv2.erode(mask, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
        trimap = np.zeros_like(mask)
        trimap[dilated == 255] = 128
        trimap[eroded == 255] = 255
        return trimap

    mask = mask.squeeze(0).cpu().detach().numpy().astype(np.uint8) * 255
    trimap = g_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1
    trimap = torch.from_numpy(trimap).unsqueeze(0)

    return tensor2pil(trimap).convert('L')


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = 'mediapipe'
    model_name = 'selfie_multiclass_256x256.tflite'

    model_file_path = ""
    try:
        model_file_path = os.path.join(os.path.normpath(folder_paths.folder_names_and_paths[model_folder_name][0][0]), model_name)
    except:
        pass
    if not os.path.exists(model_file_path):
        model_file_path = os.path.join(folder_paths.models_dir, model_folder_name, model_name)

    if not os.path.exists(model_file_path):
        import wget
        model_url = f'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}'
        log(f"Downloading '{model_name}' model")
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        wget.download(model_url, model_file_path)
    return model_file_path

def mask_fix(images:torch.Tensor, radius:int, fill_holes:int, white_threshold:float, extra_clip:float) -> torch.Tensor:
    d = radius * 2 + 1
    i_dup = copy.deepcopy(images.cpu().numpy())
    for index, image in enumerate(i_dup):
        cleaned = cv2.bilateralFilter(image, 9, 0.05, 8)
        alpha = np.clip((image - white_threshold) / (1 - white_threshold), 0, 1)
        rgb = image * alpha
        alpha = cv2.GaussianBlur(alpha, (d, d), 0) * 0.99 + np.average(alpha) * 0.01
        rgb = cv2.GaussianBlur(rgb, (d, d), 0) * 0.99 + np.average(rgb) * 0.01
        rgb = rgb / np.clip(alpha, 0.00001, 1)
        rgb = rgb * extra_clip
        cleaned = np.clip(cleaned / rgb, 0, 1)
        if fill_holes > 0:
            fD = fill_holes * 2 + 1
            gamma = cleaned * cleaned
            kD = np.ones((fD, fD), np.uint8)
            kE = np.ones((fD + 2, fD + 2), np.uint8)
            gamma = cv2.dilate(gamma, kD, iterations=1)
            gamma = cv2.erode(gamma, kE, iterations=1)
            gamma = cv2.GaussianBlur(gamma, (fD, fD), 0)
            cleaned = np.maximum(cleaned, gamma)
        i_dup[index] = cleaned
    return torch.from_numpy(i_dup)

def histogram_remap(image:torch.Tensor, blackpoint:float, whitepoint:float) -> torch.Tensor:
    bp = min(blackpoint, whitepoint - 0.001)
    scale = 1 / (whitepoint - bp)
    i_dup = copy.deepcopy(image.cpu().numpy())
    i_dup = np.clip((i_dup - bp) * scale, 0.0, 1.0)
    return torch.from_numpy(i_dup)

def expand_mask(mask:torch.Tensor, grow:int, blur:int) -> torch.Tensor:
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
        output = torch.from_numpy(output)
        out.append(output)
    # blur
    for idx, tensor in enumerate(out):
        pil_image = tensor2pil(tensor.cpu().detach())
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
        out[idx] = pil2tensor(pil_image)
    ret_mask = torch.cat(out, dim=0)
    return ret_mask

def mask_invert(mask:torch.Tensor) -> torch.Tensor:
    return 1 - mask

def subtract_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor:
    return torch.clamp(masks_a - masks_b, 0, 255)

def add_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor:
    mask = chop_image(tensor2pil(masks_a), tensor2pil(masks_b), blend_mode='add', opacity=100)
    return image2mask(mask)

def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def mask_area(image:Image) -> tuple:
    cv2_image = pil2cv2(image.convert('RGBA'))
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    locs = np.where(thresh == 255)
    try:
        x1 = np.min(locs[1])
        x2 = np.max(locs[1])
        y1 = np.min(locs[0])
        y2 = np.max(locs[0])
    except ValueError:
        x1, y1, x2, y2 = -1, -1, 0, 0
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)

def min_bounding_rect(image:Image) -> tuple:
    cv2_image = pil2cv2(image)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, 1, 2)
    x, y, width, height = 0, 0, 0, 0
    area = 0
    for contour in contours:
        _x, _y, _w, _h = cv2.boundingRect(contour)
        _area = _w * _h
        if _area > area:
            area = _area
            x, y, width, height = _x, _y, _w, _h
    return (x, y, width, height)

def max_inscribed_rect(image:Image) -> tuple:
    img = pil2cv2(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0].reshape(len(contours[0]), 2)
    rect = []
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))
    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
        while not best_rect_found and index_rect < nb_rect:
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
            valid_rect = True
            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1
            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1
            if valid_rect:
                best_rect_found = True
            index_rect += 1
    #较小的数值排前面
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)

def gray_threshold(image:Image, thresh:int=127, otsu:bool=False) -> Image:
    cv2_image = pil2cv2(image)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    if otsu:
        _, thresh =  cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_TOZERO)
    return cv22pil(thresh).convert('L')

def image_to_colormap(image:Image, index:int) -> Image:
    return cv22pil(cv2.applyColorMap(pil2cv2(image), index))

# 检查mask有效区域面积比例
def mask_white_area(mask:Image, white_point:int) -> float:
    if mask.mode != 'L':
        mask.convert('L')
    white_pixels = 0
    for y in range(mask.height):
        for x in range(mask.width):
            mask.getpixel((x, y)) > 16
            if mask.getpixel((x, y)) > white_point:
                white_pixels += 1
    return white_pixels / (mask.width * mask.height)

'''Color Functions'''

def color_balance(image:Image, shadows:list, midtones:list, highlights:list,
                  shadow_center:float=0.15, midtone_center:float=0.5, highlight_center:float=0.8,
                  shadow_max:float=0.1, midtone_max:float=0.3, highlight_max:float=0.2,
                  preserve_luminosity:bool=False) -> Image:

    img = pil2tensor(image)
    # Create a copy of the img tensor
    img_copy = img.clone()

    # Calculate the original luminance if preserve_luminosity is True
    if preserve_luminosity:
        original_luminance = 0.2126 * img_copy[..., 0] + 0.7152 * img_copy[..., 1] + 0.0722 * img_copy[..., 2]

    # Define the adjustment curves
    def adjust(x, center, value, max_adjustment):
        # Scale the adjustment value
        value = value * max_adjustment

        # Define control points
        points = torch.tensor([[0, 0], [center, center + value], [1, 1]])

        # Create cubic spline
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(points[:, 0], points[:, 1])

        # Apply the cubic spline to the color channel
        return torch.clamp(torch.from_numpy(cs(x)), 0, 1)

    # Apply the adjustments to each color channel
    # shadows, midtones, highlights are lists of length 3 (for R, G, B channels) with values between -1 and 1
    for i, (s, m, h) in enumerate(zip(shadows, midtones, highlights)):
        img_copy[..., i] = adjust(img_copy[..., i], shadow_center, s, shadow_max)
        img_copy[..., i] = adjust(img_copy[..., i], midtone_center, m, midtone_max)
        img_copy[..., i] = adjust(img_copy[..., i], highlight_center, h, highlight_max)

    # If preserve_luminosity is True, adjust the RGB values to match the original luminance
    if preserve_luminosity:
        current_luminance = 0.2126 * img_copy[..., 0] + 0.7152 * img_copy[..., 1] + 0.0722 * img_copy[..., 2]
        img_copy *= (original_luminance / current_luminance).unsqueeze(-1)

    return tensor2pil(img_copy)

def RGB_to_Hex(RGB:tuple) -> str:
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def Hex_to_RGB(inhex:str) -> tuple:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code in {inhex}')
    else:
        if len(inhex) == 4:
            inhex = "#" + "".join([char * 2 for char in inhex[1:]])
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)

def RGB_to_HSV(RGB:tuple) -> list:
    HSV = rgb_to_hsv(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0)
    return [int(x * 360) for x in HSV]

def Hex_to_HSV_255level(inhex:str) -> list:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code in {inhex}')
    else:
        if len(inhex) == 4:
            inhex = "#" + "".join([char * 2 for char in inhex[1:]])
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        RGB = (int(rval, 16), int(gval, 16), int(bval, 16))
        HSV = rgb_to_hsv(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0)
    return [int(x * 255) for x in HSV]

def HSV_255level_to_Hex(HSV: list) -> str:
    if len(HSV) != 3 or any((not isinstance(v, int) or v < 0 or v > 255) for v in HSV):
        raise ValueError('Invalid HSV values, each value should be an integer between 0 and 255')

    H, S, V = HSV
    RGB = tuple(int(x * 255) for x in hsv_to_rgb(H / 255.0, S / 255.0, V / 255.0))

    # Convert RGB values to hexadecimal format
    hex_r = format(RGB[0], '02x')
    hex_g = format(RGB[1], '02x')
    hex_b = format(RGB[2], '02x')

    return '#' + hex_r + hex_g + hex_b

# 返回补色色值
def complementary_color(color: str) -> str:
    color = Hex_to_RGB(color)
    return RGB_to_Hex((255 - color[0], 255 - color[1], 255 - color[2]))

# 返回颜色对应灰度值
def rgb2gray(color:str)->int:
    (r, g, b) = Hex_to_RGB(color)
    return int((r * 299 + g * 587 + b * 114) / 1000)

'''Value Functions'''
def is_valid_mask(tensor:torch.Tensor) -> bool:
    return not bool(torch.all(tensor == 0).item())

def step_value(start_value, end_value, total_step, step) -> float:  # 按当前步数在总步数中的位置返回比例值
    factor = step / total_step
    return (end_value - start_value) * factor + start_value

def step_color(start_color_inhex:str, end_color_inhex:str, total_step:int, step:int) -> str:  # 按当前步数在总步数中的位置返回比例颜色
    start_color = tuple(Hex_to_RGB(start_color_inhex))
    end_color = tuple(Hex_to_RGB(end_color_inhex))
    start_R, start_G, start_B = start_color[0], start_color[1], start_color[2]
    end_R, end_G, end_B = end_color[0], end_color[1], end_color[2]
    ret_color = (int(step_value(start_R, end_R, total_step, step)),
                 int(step_value(start_G, end_G, total_step, step)),
                 int(step_value(start_B, end_B, total_step, step)),
                 )
    return RGB_to_Hex(ret_color)

def has_letters(string:str) -> bool:
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    if match:
        return True
    else:
        return False


def replace_case(old:str, new:str, text:str) -> str:
    index = text.lower().find(old.lower())
    if index == -1:
        return text
    return replace_case(old, new, text[:index] + new + text[index + len(old):])

def random_numbers(total:int, random_range:int, seed:int=0, sum_of_numbers:int=0) -> list:
    random.seed(seed)
    numbers = [random.randint(-random_range//2, random_range//2) for _ in range(total - 1)]
    avg = sum(numbers) // total
    ret_list = []
    for i in numbers:
        ret_list.append(i - avg)
    ret_list.append((sum_of_numbers - sum(ret_list)) // 2)
    return ret_list

# 四舍五入取整数倍
def num_round_to_multiple(number:int, multiple:int) -> int:
    remainder = number % multiple
    if remainder == 0 :
        return number
    else:
        factor = int(number / multiple)
        if number - factor * multiple > multiple / 2:
            factor += 1
        return factor * multiple

# 向上取整数倍
def num_round_up_to_multiple(number: int, multiple: int) -> int:
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        factor = (number + multiple - 1) // multiple  # 向上取整的计算方式
        return factor * multiple

def calculate_side_by_ratio(orig_width:int, orig_height:int, ratio:float, longest_side:int=0) -> int:

    if orig_width > orig_height:
        if longest_side:
            target_width = longest_side
        else:
            target_width = orig_width
        target_height = int(target_width / ratio)
    else:
        if longest_side:
            target_height = longest_side
        else:
            target_height = orig_height
        target_width = int(target_height * ratio)

    if ratio < 1:
        if longest_side:
            _r = longest_side / target_height
            target_height = longest_side
        else:
            _r = orig_height / target_height
            target_height = orig_height
        target_width = int(target_width * _r)

    return target_width, target_height

def generate_random_name(prefix:str, suffix:str, length:int) -> str:
    name = ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(length))
    return prefix + name + suffix

def check_image_file(file_name:str, interval:int) -> object:
    while True:
        if os.path.isfile(file_name):
            try:
                image = Image.open(file_name)
                ret_image = copy.deepcopy(image)
                image.close()
                return ret_image
            except Exception as e:
                log(e)
                return None
            break
        time.sleep(interval / 1000)

# 判断字符串是否包含中文
def is_contain_chinese(check_str:str) -> bool:
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

# 生成随机颜色
def generate_random_color():
    """
    Generate a random color in hexadecimal format.
    """
    # random.seed(int(time.time()))
    return "#{:06x}".format(random.randint(0x101010, 0xFFFFFF))

# 提取字符串中的int数为列表
def extract_numbers(string):
    return [int(s) for s in re.findall(r'\d+', string)]

# 提取字符串中的数值, 返回为列表
def extract_all_numbers_from_str(string, checkint:bool=False):
    # 定义浮点数的正则表达式模式
    number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    # 使用re.findall找到所有匹配的字符串
    matches = re.findall(number_pattern, string)
    # 转换为浮点数
    numbers = [float(match) for match in matches]
    number_list = []
    # 如果需要检查是否为整数，则将浮点数转换为整数
    if checkint:
        for num in numbers:
            int_num = int(num)
            if math.isclose(num, int_num, rel_tol=1e-19):
                number_list.append(int_num)
            else:
                number_list.append(num)
    else:
        number_list = numbers

    return number_list



# 提取字符串中用"," ";" " "分开的字符串, 返回为列表
def extract_substr_from_str(string) -> list:
    return re.split(r'[,\s;，；]+', string)

# lcs匹配算法，计算最长公共子序列 (LCS)：子字符串顺序：以相同顺序出现，权重更高。额外字符惩罚：多余字符会降低相似度。
def lcs_with_order(s1, s2):
    """Calculate the length of the longest common subsequence (LCS) with the same order."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# 使用正则表达式将字符串拆分为单词（token），对比的同时忽略大小写和非字母数字字符。
def tokenize_string(s):
    """Tokenize a string by splitting on non-alphanumeric characters and normalizing case."""
    return re.findall(r'\b\w+\b', s.lower())

# 在列表中找到字符串的最佳匹配
def find_best_match_by_similarity(target, candidates):
    """
    Find the best matching string based on substring order, extra character penalties, and tokenization.

    Parameters:
        target (str): The target string.
        candidates (list of str): List of candidate strings.

    Returns:
        str: The best matching string.
    """
    target_tokens = tokenize_string(target)
    best_match = None
    highest_score = float('-inf')

    for candidate in candidates:
        candidate_tokens = tokenize_string(candidate)

        # Calculate LCS on tokens
        target_str = ''.join(target_tokens)
        candidate_str = ''.join(candidate_tokens)
        lcs = lcs_with_order(target_str, candidate_str)

        # Calculate similarity score
        match_ratio = lcs / len(target_str)  # Ratio of matched characters
        extra_char_penalty = len(candidate_str) - lcs  # Penalty for extra characters
        unmatched_tokens_penalty = len(set(candidate_tokens) - set(target_tokens))  # Penalty for unmatched tokens
        score = match_ratio - 0.1 * extra_char_penalty - 0.2 * unmatched_tokens_penalty  # Weighted score

        if score > highest_score:
            highest_score = score
            best_match = candidate

    return best_match


def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def tensor_info(tensor:object) -> str:
    value = ''
    if isinstance(tensor, torch.Tensor):
        value += f"\n Input dim = {tensor.dim()}, shape[0] = {tensor.shape[0]} \n"
        for i in range(tensor.shape[0]):
            t = tensor[i]
            image = tensor2pil(t)
            value += f'\n index {i}: Image.size = {image.size}, Image.mode = {image.mode}, dim = {t.dim()}, '
            for j in range(t.dim()):
                value += f'shape[{j}] = {t.shape[j]}, '
    else:
        value = f"tensor_info: Not tensor, type is {type(tensor)}"
    return value

# 去除空行
def remove_empty_lines(text):
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines)

# 去除重复的句子
def remove_duplicate_string(text:str) -> str:
    sentences = re.split(r'(?<=[:;,.!?])\s+', text)
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

files_for_uform_gen2_qwen = Path(os.path.join(folder_paths.models_dir, "LLavacheckpoints", "files_for_uform_gen2_qwen"))
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [151645]  # Define stop tokens as per your model's specifics
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class UformGen2QwenChat:

    def __init__(self):
        from huggingface_hub import snapshot_download
        # self.model_path = snapshot_download("unum-cloud/uform-gen2-qwen-500m",
        #                                     local_dir=files_for_uform_gen2_qwen,
        #                                     force_download=False,  # Set to True if you always want to download, regardless of local copy
        #                                     local_files_only=False,  # Set to False to allow downloading if not available locally
        #                                     local_dir_use_symlinks="auto") # or set to True/False based on your symlink preference
        self.model_path = files_for_uform_gen2_qwen
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def chat_response(self, message, history, image_path):
        stop = StopOnTokens()
        messages = [{"role": "system", "content": "You are a helpful Assistant."}]

        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        if len(messages) == 1:
            message = f" <image>{message}"

        messages.append({"role": "user", "content": message})

        model_inputs = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        image = Image.open(image_path)  # Load image using PIL
        image_tensor = (
            self.processor.feature_extractor(image)
            .unsqueeze(0)
        )

        attention_mask = torch.ones(
            1, model_inputs.shape[1] + self.processor.num_image_latents - 1
        )

        model_inputs = {
            "input_ids": model_inputs,
            "images": image_tensor,
            "attention_mask": attention_mask
        }

        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.2,
                stopping_criteria=StoppingCriteriaList([stop])
            )

        response_text = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
        response_text = remove_duplicate_string(response_text)
        return response_text

'''CLASS'''

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
  def __eq__(self, __value: object) -> bool:
    return True
  def __ne__(self, __value: object) -> bool:
    return False



'''Load File'''

def download_hg_model(model_id:str,exDir:str='') -> str:
    # 下载本地
    model_checkpoint = os.path.join(folder_paths.models_dir, exDir, os.path.basename(model_id))
    if not os.path.exists(model_checkpoint):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
    return model_checkpoint


def get_files(model_path: str, file_ext_list:list) -> dict:
    file_list = []
    for ext in file_ext_list:
        file_list.extend(glob.glob(os.path.join(model_path, '*' + ext)))
    files_dict = {}
    for i in range(len(file_list)):
        _, filename = os.path.split(file_list[i])
        files_dict[filename] = file_list[i]
    return files_dict

# def load_inference_prompt() -> str:
#     inference_prompt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "resource",
#                                          "inference.prompt")
#     ret_value = ''
#     try:
#         with open(inference_prompt_file, 'r') as f:
#             ret_value = f.readlines()
#     except Exception as e:
#         log(f'Warning: {inference_prompt_file} ' + repr(e) + f", check it to be correct. ", message_type='warning')
#     return  ''.join(ret_value)

def load_custom_size() -> list:
    custom_size_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "custom_size.ini")
    ret_value = ['1024 x 1024',
                '768 x 512',
                '512 x 768',
                '1280 x 720',
                '720 x 1280',
                '1344 x 768',
                '768 x 1344',
                '1536 x 640',
                '640 x 1536'
                 ]
    try:
        with open(custom_size_file, 'r') as f:
            ini = f.readlines()
            for line in ini:
                if not line.startswith(f'#'):
                    ret_value.append(line.strip())
    except Exception as e:
        pass
        # log(f'Warning: {custom_size_file} not found' + f", use default size. ")
    return ret_value

def get_api_key(api_name:str) -> str:
    api_key_ini_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "api_key.ini")
    ret_value = ''
    try:
        with open(api_key_ini_file, 'r') as f:
            ini = f.readlines()
            for line in ini:
                if line.startswith(f'{api_name}='):
                    ret_value = line[line.find('=') + 1:].rstrip().lstrip()
                    break
    except Exception as e:
        log(f'Warning: {api_key_ini_file} ' + repr(e) + f", check it to be correct. ", message_type='warning')
    remove_char = ['"', "'", '“', '”', '‘', '’']
    for i in remove_char:
        if i in ret_value:
            ret_value = ret_value.replace(i, '')
    if len(ret_value) < 4:
        log(f'Warning: Invalid API-key, Check the key in {api_key_ini_file}.', message_type='warning')
    return ret_value

# 判断文件名后缀是否包括在列表中(忽略大小写)
def file_is_extension(filename:str, ext_list:tuple) -> bool:
    # 获取文件的真实后缀（包括点）
    true_ext = os.path.splitext(filename)[1]
    if true_ext.lower() in ext_list:
        return True
    return False

# 遍历目录下包括子目录指定后缀文件，返回字典
def collect_files(root_dir:str, suffixes:tuple, default_dir:str=""):
    result = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file_is_extension(file, suffixes):
                # 获取文件的完整路径作为 value
                full_path = os.path.join(dirpath, file)
                # 如果是default_dir 则去掉路径，使用文件名作为 key
                if dirpath == default_dir:
                    relative_path = os.path.relpath(full_path, root_dir)
                    result.update({relative_path: full_path})
                else:
                    result.update({full_path: full_path})
    return result


def get_resource_dir() -> list:
    default_lut_dir = []
    default_lut_dir.append(os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'lut'))
    default_font_dir = []
    default_font_dir.append(os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'font'))
    resource_dir_ini_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))),
                                         "resource_dir.ini")
    try:
        with open(resource_dir_ini_file, 'r') as f:
            ini = f.readlines()
            for line in ini:
                if line.startswith('LUT_dir='):
                    _ldir = line[line.find('=') + 1:].rstrip().lstrip()
                    for dir in extract_substr_from_str(_ldir) :
                        if os.path.exists(dir):
                            default_lut_dir.append(dir)
                elif line.startswith('FONT_dir='):
                    _fdir = line[line.find('=') + 1:].rstrip().lstrip()
                    for dir in extract_substr_from_str(_fdir):
                        if os.path.exists(dir):
                            default_font_dir.append(dir)
    except Exception as e:
        pass
        # log(f'Warning: {resource_dir_ini_file} not found' + f", default directory to be used. ")


    LUT_DICT = {}
    for dir in default_lut_dir:
        LUT_DICT.update(collect_files(root_dir=dir, suffixes= ('.cube'), default_dir=default_lut_dir[0] )) # 后缀要小写
    LUT_LIST = list(LUT_DICT.keys())

    FONT_DICT = {}
    for dir in default_font_dir:
        FONT_DICT.update(collect_files(root_dir=dir, suffixes=('.ttf', '.otf'), default_dir=default_font_dir[0])) # 后缀要小写
    FONT_LIST = list(FONT_DICT.keys())

    return (LUT_DICT, FONT_DICT)

# 规范bbox，保证x1 < x2, y1 < y2, 并返回int
def standardize_bbox(bboxes:list) -> list:
    ret_bboxes = []
    for bbox in bboxes:
        x1 = int(min(bbox[0], bbox[2]))
        y1 = int(min(bbox[1], bbox[3]))
        x2 = int(max(bbox[0], bbox[2]))
        y2 = int(max(bbox[1], bbox[3]))
        ret_bboxes.append([x1, y1, x2, y2])
    return ret_bboxes

def draw_bounding_boxes(image: Image, bboxes: list, color: str = "#FF0000", line_width: int = 5) -> Image:
    """
    Draw bounding boxes on the image using the coordinates provided in the bboxes dictionary.
    """

    (_, FONT_DICT) = get_resource_dir()

    font_size = 25
    font = ImageFont.truetype(list(FONT_DICT.items())[0][1], font_size)

    if len(bboxes) > 0:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        if line_width < 0:  # auto line width
            line_width = (image.width + image.height) // 1000

        for index, box in enumerate(bboxes):
            random_color = generate_random_color()
            if color != "random":
                random_color = color
            xmin = min(box[0], box[2])
            xmax = max(box[0], box[2])
            ymin = min(box[1], box[3])
            ymax = max(box[1], box[3])
            draw.rectangle([xmin, ymin, xmax, ymax], outline=random_color, width=line_width)
            draw.text((xmin, ymin - font_size*1.2), str(index), font=font, fill=random_color)

    return image

def draw_bbox(image: Image, bbox: tuple, color: str = "#FF0000", line_width: int = 5, title: str = "", font_size: int = 10) -> Image:
    """
    Draw bounding boxes on the image using the coordinates provided in the bboxes dictionary.
    """

    (_, FONT_DICT) = get_resource_dir()

    font = ImageFont.truetype(list(FONT_DICT.items())[0][1], font_size)

    draw = ImageDraw.Draw(image)
    width, height = image.size
    if line_width < 0:  # auto line width
        line_width = (image.width + image.height) // 1000

    random_color = generate_random_color()
    if color != "random":
        random_color = color
    xmin = min(bbox[0], bbox[2])
    xmax = max(bbox[0], bbox[2])
    ymin = min(bbox[1], bbox[3])
    ymax = max(bbox[1], bbox[3])
    draw.rectangle([xmin, ymin, xmax, ymax], outline=random_color, width=line_width)
    if title != "":
        draw.text((xmin, ymin - font_size*1.2), title, font=font, fill=random_color)

    return image



'''Constant'''

chop_mode = [
    'normal',
    'multply',
    'screen',
    'add',
    'subtract',
    'difference',
    'darker',
    'lighter',
    'color_burn',
    'color_dodge',
    'linear_burn',
    'linear_dodge',
    'overlay',
    'soft_light',
    'hard_light',
    'vivid_light',
    'pin_light',
    'linear_light',
    'hard_mix'
    ]

# Blend Mode from Virtuoso Pack https://github.com/chrisfreilich/virtuoso-nodes
chop_mode_v2 = list(BLEND_MODES.keys())

gemini_generate_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 400
}

gemini_safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

minicpm_llama3_v25_prompts = """
        # MISSION
        You are an imagine generator for a slide deck tool. You will be given the text or description of a slide and you'll generate a few image descriptions that will be fed to an AI image generator. It will need to have a particular format (seen below). You will also be given some examples below. Think metaphorically and symbolically. 

        # FORMAT
        The format should follow this general pattern:

        <MAIN SUBJECT>, <DESCRIPTION OF MAIN SUBJECT>, <BACKGROUND OR CONTEXT, LOCATION, ETC>, <STYLE, GENRE, MOTIF, ETC>, <COLOR SCHEME>, <CAMERA DETAILS>

        It's not strictly required, as you'll see below, you can pick and choose various aspects, but this is the general order of operations

        # EXAMPLES

        a Shakespeare stage play, yellow mist, atmospheric, set design by Michel Crête, Aerial acrobatics design by André Simard, hyperrealistic, 4K, Octane render, unreal engine

        The Moon Knight dissolving into swirling sand, volumetric dust, cinematic lighting, close up portrait

        ethereal Bohemian Waxwing bird, Bombycilla garrulus :: intricate details, ornate, detailed illustration, octane render :: Johanna Rupprecht style, William Morris style :: trending on artstation

        steampunk cat, octane render, hyper realistic

        Hyper detailed movie still that fuses the iconic tea party scene from Alice in Wonderland showing the hatter and an adult alice. a wooden table is filled with teacups and cannabis plants. The scene is surrounded by flying weed. Some playcards flying around in the air. Captured with a Hasselblad medium format camera

        venice in a carnival picture 3, in the style of fantastical compositions, colorful, eye-catching compositions, symmetrical arrangements, navy and aquamarine, distinctive noses, gothic references, spiral group –style expressive

        Beautiful and terrifying Egyptian mummy, flirting and vamping with the viewer, rotting and decaying climbing out of a sarcophagus lunging at the viewer, symmetrical full body Portrait photo, elegant, highly detailed, soft ambient lighting, rule of thirds, professional photo HD Photography, film, sony, portray, kodak Polaroid 3200dpi scan medium format film Portra 800, vibrantly colored portrait photo by Joel – Peter Witkin + Diane Arbus + Rhiannon + Mike Tang, fashion shoot

        A grandmotherly Fate sits on a cozy cosmic throne knitting with mirrored threads of time, the solar system spins like clockwork behind her as she knits the futures of people together like an endless collage of destiny, maximilism, cinematic quality, sharp – focus, intricate details

        A cloud with several airplanes flying around on top, in the style of detailed fantasy art, nightcore, quiet moments captured in paint, radiant clusters, i cant believe how beautiful this is, detailed character design, dark cyan and light crimson

        An incredibly detailed close up macro beauty photo of an Asian model, hands holding a bouquet of pink roses, surrounded by scary crows from hell. Shot on a Hasselblad medium format camera with a 100mm lens. Unmistakable to a photograph. Cinematic lighting. Photographed by Tim Walker, trending on 500px

        Game-Art | An island with different geographical properties and multiple small cities floating in space ::10 Island | Floating island in space – waterfalls over the edge of the island falling into space – island fragments floating around the edge of the island, Mountain Ranges – Deserts – Snowy Landscapes – Small Villages – one larger city ::8 Environment | Galaxy – in deep space – other universes can be seen in the distance ::2 Style | Unreal Engine 5 – 8K UHD – Highly Detailed – Game-Art

        a warrior sitting on a giant creature and riding it in the water, with wings spread wide in the water, camera positioned just above the water to capture this beautiful scene, surface showing intricate details of the creature’s scales, fins, and wings, majesty, Hero rides on the creature in the water, digitally enhanced, enhanced graphics, straight, sharp focus, bright lighting, closeup, cinematic, Bronze, Azure, blue, ultra highly detailed, 18k, sharp focus, bright photo with rich colors, full coverage of a scene, straight view shot

        A real photographic landscape painting with incomparable reality,Super wide,Ominous sky,Sailing boat,Wooden boat,Lotus,Huge waves,Starry night,Harry potter,Volumetric lighting,Clearing,Realistic,James gurney,artstation

        Tiger monster with monstera plant over him, back alley in Bangkok, art by Otomo Katsuhiro crossover Yayoi Kusama and Hayao Miyazaki

        An elderly Italian woman with wrinkles, sitting in a local cafe filled with plants and wood decorations, looking out the window, wearing a white top with light purple linen blazer, natural afternoon light shining through the window

        # OUTPUT
        Your output should just be an plain list of descriptions. No numbers, no extraneous labels, no hyphens.
        Create only one prompt.
        """
