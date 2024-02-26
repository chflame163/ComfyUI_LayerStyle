'''Image process functions for ComfyUI nodes
by chflame https://github.com/chflame163
'''
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
from typing import Union, List
from PIL import Image, ImageFilter, ImageChops, ImageDraw, ImageOps, ImageEnhance, ImageFont
from skimage import img_as_float, img_as_ubyte
from pymatting import fix_trimap, estimate_alpha_cf, estimate_foreground_ml
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import colorsys
from colour.io.luts.iridas_cube import read_LUT_IridasCube, LUT3D, LUT3x1D
from typing import Union
import folder_paths as COMFY_FOLDER_PATH
from .briarmbg import BriaRMBG
from .filmgrainer import processing as processing_utils
from .filmgrainer import filmgrainer as filmgrainer

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
    print(e)
    log(f'Dependency package error, unable import "cv2.ximgproc".'
        f'\nPlease REINSTALL package "opencv-contrib-python".'
        f'\nFor detail refer to \033[4mhttps://github.com/chflame163/ComfyUI_LayerStyle/issues/5\033[0m',
        message_type='error')


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
    ret_mask = torch.tensor([pil2tensor(_image)[0, :, :, 3].tolist()])
    return ret_mask

def mask2image(mask:torch.Tensor)  -> Image:
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color='white')
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image

# def make_3d_mask(mask):
#     if len(mask.shape) == 4:
#         return mask.squeeze(0)
#     elif len(mask.shape) == 2:
#         return mask.unsqueeze(0)
#     return mask

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

def remove_background(image:Image, mask:Image, color:str) -> Image:
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=color)
    ret_image.paste(image, mask=mask)
    return ret_image

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

def filmgrain_image(image:Image, scale:float, grain_power:float,
                    shadows:float, highs:float, grain_sat:float,
                    sharpen:int=1, grain_type:int=4, src_gamma:float=1.0,
                    gray_scale:bool=False, seed:int=0) -> Image:
    # image = pil2tensor(image)
    # grain_type, 1=fine, 2=fine simple, 3=coarse, 4=coarser
    grain_type_index = 3

    # Apply grain
    grain_image = filmgrainer.process(image, scale=scale, src_gamma=src_gamma, grain_power=grain_power,
                                      shadows=shadows, highs=highs, grain_type=grain_type_index,
                                      grain_sat=grain_sat, gray_scale=gray_scale, sharpen=sharpen, seed=seed)
    return tensor2pil(torch.from_numpy(grain_image).unsqueeze(0))

def __apply_radialblur(image, blur_strength, radial_mask, focus_spread, steps):
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
    return (_image, _alpha, ret_image)


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

def draw_rect(image:Image, x:int, y:int, width:int, height:int, line_color:str, line_width:int,
              box_color:str=None) -> Image:
    # image = image.convert('RGBA')
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + width, y + height), fill=box_color, outline=line_color, width=line_width, )
    return image

def draw_border(image:Image, border_width:int, color:str='#FFFFFF') -> Image:
    return ImageOps.expand(image, border=border_width, fill=color)

def get_image_color_tone(image:Image) -> str:
    image = image.convert('RGB')
    max_score = 0.0001
    dominant_color = None
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13,235)
        y = (y - 16.0) / (235 - 16)
        if y>0.9:
            continue
        score = (saturation+0.1)*count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
        ret_color = RGB_to_Hex(dominant_color)
    return ret_color

def get_image_color_average(image:Image) -> str:
    image = image.convert('RGB')
    width, height = image.size
    total_red = 0
    total_green = 0
    total_blue = 0
    for y in range(height):
        for x in range(width):
            rgb = image.getpixel((x, y))
            total_red += rgb[0]
            total_green += rgb[1]
            total_blue += rgb[2]

    average_red = total_red // (width * height)
    average_green = total_green // (width * height)
    average_blue = total_blue // (width * height)
    color = (average_red, average_green, average_blue)
    ret_color = RGB_to_Hex(color)
    return ret_color

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
    width = image.width
    height = image.height
    ret_image = Image.new('L', size=(width, height), color='black')
    for x in range(width):
        for y in range(height):
                pixel = image.getpixel((x, y))
                _pixel = pixel + offset
                if _pixel > 255:
                    _pixel = 255
                if _pixel < 0:
                    _pixel = 0
                ret_image.putpixel((x, y), _pixel)
    return ret_image

def image_hue_offset(image:Image, offset:int) -> Image:
    image = image.convert('L')
    width = image.width
    height = image.height
    ret_image = Image.new('L', size=(width, height), color='black')
    for x in range(width):
        for y in range(height):
                pixel = image.getpixel((x, y))
                _pixel = pixel + offset
                if _pixel > 255:
                    _pixel -= 256
                if _pixel < 0:
                    _pixel += 256
                ret_image.putpixel((x, y), _pixel)
    return ret_image

def gamma_trans(image:Image, gamma:float) -> Image:
    cv2_image = pil2cv2(image)
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    _corrected = cv2.LUT(cv2_image,gamma_table)
    return cv22pil(_corrected)

def apply_lut(image:Image, lut_file:str, log:bool=False) -> Image:
    lut: Union[LUT3x1D, LUT3D] = read_LUT_IridasCube(lut_file)
    lut.name = os.path.splitext(os.path.basename(lut_file))[0]  # use base filename instead of internal LUT name

    im_array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
    dom_scale = None
    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        im_array = im_array * dom_scale + lut.domain[0]
    if log:
        im_array = im_array ** (1 / 2.2)
    im_array = lut.apply(im_array)
    if log:
        im_array = im_array ** (2.2)
    if is_non_default_domain:
        im_array = (im_array - lut.domain[0]) / dom_scale
    im_array = im_array * 255
    ret_image = Image.fromarray(np.uint8(im_array))

    return ret_image


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
    sigmaColor = int((image.width + image.height) / 256 * factor)
    sigmaSpace = int((image.width + image.height) / 160 * factor)
    img_bit = cv2.bilateralFilter(src=img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    ret_image = cv2.cvtColor(img_bit, cv2.COLOR_BGR2RGB)
    return cv22pil(ret_image)


def pixel_spread(image:Image, mask:Image) -> Image:
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

'''Mask Functions'''

def load_RMBG_model():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = BriaRMBG()
    model_path = ""
    try:
        model_path = os.path.join(os.path.normpath(COMFY_FOLDER_PATH.folder_names_and_paths['rmbg'][0][0]), "model.pth")
    except:
        pass
    if not os.path.exists(model_path):
        model_path = os.path.join(COMFY_FOLDER_PATH.models_dir, "rmbg", "RMBG-1.4", "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(current_directory), "RMBG-1.4", "model.pth")
    net.load_state_dict(torch.load(model_path, map_location=device))
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
    _mask = Image.fromarray(np.squeeze(im_array)).convert('L')
    return _mask

def mask_edge_detail(image:torch.Tensor, mask:torch.Tensor, detail_range:int=8, black_point:float=0.01, white_point:float=0.99) -> torch.Tensor:
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

def RGB2RGBA(image:Image, mask:Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

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

'''Color Functions'''

def RGB_to_Hex(RGB) -> str:
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

def Hex_to_RGB(inhex) -> tuple:
    rval = inhex[1:3]
    gval = inhex[3:5]
    bval = inhex[5:]
    rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)

def RGB_to_HSV(RGB:tuple) -> list:
    HSV = colorsys.rgb_to_hsv(RGB[0] / 255.0, RGB[1] / 255.0, RGB[2] / 255.0)
    return [int(x * 360) for x in HSV]

'''Value Functions'''

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

def random_numbers(total:int, random_range:int, seed:int=0, sum_of_numbers:int=0) -> list:
    random.seed(seed)
    numbers = [random.randint(-random_range//2, random_range//2) for _ in range(total - 1)]
    avg = sum(numbers) // total
    ret_list = []
    for i in numbers:
        ret_list.append(i - avg)
    ret_list.append((sum_of_numbers - sum(ret_list)) // 2)
    return ret_list

def num_round_to_multiple(number:int, multiple:int) -> int:
    remainder = number % multiple
    if remainder == 0 :
        return number
    else:
        factor = int(number / multiple)
        if number - factor * multiple > multiple / 2:
            factor += 1
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
'''CLASS'''

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
  def __ne__(self, __value: object) -> bool:
    return False

'''Constant'''

chop_mode = ['normal', 'multply', 'screen', 'add', 'subtract', 'difference', 'darker', 'lighter',
             'color_burn', 'color_dodge', 'linear_burn', 'linear_dodge', 'overlay',
             'soft_light', 'hard_light', 'vivid_light', 'pin_light', 'linear_light', 'hard_mix']

'''Load INI File'''

default_lut_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'lut')
default_font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'font')
resource_dir_ini_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "resource_dir.ini")

try:
    with open(resource_dir_ini_file, 'r') as f:
        ini = f.readlines()
        for line in ini:
            if line.startswith('LUT_dir='):
                _ldir = line[line.find('=') + 1:].rstrip().lstrip()
                if os.path.exists(_ldir):
                    default_lut_dir = _ldir
                else:
                    log(f'Invalid LUT directory, default to be used. check {resource_dir_ini_file}')
            elif line.startswith('FONT_dir='):
                _fdir = line[line.find('=') + 1:].rstrip().lstrip()
                if os.path.exists(_fdir):
                    default_font_dir = _fdir
                else:
                    log(f'Invalid FONT directory, default to be used. check {resource_dir_ini_file}')
except Exception as e:
    log(f'Warning: {resource_dir_ini_file} ' + repr(e) + f", default directory to be used. ", message_type='warning')

__lut_file_list = glob.glob(default_lut_dir + '/*.cube')
LUT_DICT = {}
for i in range(len(__lut_file_list)):
    _, __filename =  os.path.split(__lut_file_list[i])
    LUT_DICT[__filename] = __lut_file_list[i]
LUT_LIST = list(LUT_DICT.keys())
log(f'Find {len(LUT_LIST)} LUTs in {default_lut_dir}')

__font_file_list = glob.glob(default_font_dir + '/*.ttf')
__font_file_list.extend(glob.glob(default_font_dir + '/*.otf'))
FONT_DICT = {}
for i in range(len(__font_file_list)):
    _, __filename =  os.path.split(__font_file_list[i])
    FONT_DICT[__filename] = __font_file_list[i]
FONT_LIST = list(FONT_DICT.keys())
log(f'Find {len(FONT_LIST)} Fonts in {default_font_dir}')
