"""
author: Chris Freilich
description: This extension provides a blend modes node with 30 blend modes.
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
# from colorsys import rgb_to_hsv, hsv_to_rgb
from skimage.color import rgb2hsv, hsv2rgb
from blend_modes import difference, normal, screen, soft_light, lighten_only, dodge,   \
                        addition, darken_only, multiply, hard_light,  \
                        grain_extract, grain_merge, divide, overlay

def dissolve(backdrop, source, opacity):
    # Normalize the RGB and alpha values to 0-1
    backdrop_norm = backdrop[:, :, :3] / 255
    source_norm = source[:, :, :3] / 255
    source_alpha_norm = source[:, :, 3] / 255

    # Calculate the transparency of each pixel in the source image
    transparency = opacity * source_alpha_norm

    # Generate a random matrix with the same shape as the source image
    random_matrix = np.random.random(source.shape[:2])

    # Create a mask where the random values are less than the transparency
    mask = random_matrix < transparency

    # Use the mask to select pixels from the source or backdrop
    blend = np.where(mask[..., None], source_norm, backdrop_norm)

    # Apply the alpha channel of the source image to the blended image
    new_rgb = (1 - source_alpha_norm[..., None]) * backdrop_norm + source_alpha_norm[..., None] * blend

    # Ensure the RGB values are within the valid range
    new_rgb = np.clip(new_rgb, 0, 1)

    # Convert the RGB values back to 0-255
    new_rgb = new_rgb * 255

    # Calculate the new alpha value by taking the maximum of the backdrop and source alpha channels
    new_alpha = np.maximum(backdrop[:, :, 3], source[:, :, 3])

    # Create a new RGBA image with the calculated RGB and alpha values
    result = np.dstack((new_rgb, new_alpha))

    return result

def hsv(backdrop, source, opacity, channel):
    # Convert RGBA to RGB, normalized
    backdrop_rgb = backdrop[:, :, :3] / 255.0
    source_rgb = source[:, :, :3] / 255.0
    source_alpha = source[:, :, 3] / 255.0

    # Convert RGB to HSV
    # backdrop_hsv = np.array([rgb_to_hsv(*rgb) for row in backdrop_rgb for rgb in row]).reshape(backdrop.shape[:2] + (3,))
    # source_hsv = np.array([rgb_to_hsv(*rgb) for row in source_rgb for rgb in row]).reshape(source.shape[:2] + (3,))
    backdrop_hsv = rgb2hsv(backdrop_rgb)
    source_hsv = rgb2hsv(source_rgb)
    
    # Combine HSV values
    new_hsv = backdrop_hsv.copy()
    
    # Determine which channel to operate on
    if channel == "saturation":
        new_hsv[:, :, 1] = (1 - opacity * source_alpha) * backdrop_hsv[:, :, 1] + opacity * source_alpha * source_hsv[:, :, 1]
    elif channel == "luminance":
        new_hsv[:, :, 2] = (1 - opacity * source_alpha) * backdrop_hsv[:, :, 2] + opacity * source_alpha * source_hsv[:, :, 2]
    elif channel == "hue":
        new_hsv[:, :, 0] = (1 - opacity * source_alpha) * backdrop_hsv[:, :, 0] + opacity * source_alpha * source_hsv[:, :, 0]
    elif channel == "color":
        new_hsv[:, :, :2] = (1 - opacity * source_alpha[..., None]) * backdrop_hsv[:, :, :2] + opacity * source_alpha[..., None] * source_hsv[:, :, :2]

    # Convert HSV back to RGB
    # new_rgb = np.array([hsv_to_rgb(*hsv) for row in new_hsv for hsv in row]).reshape(backdrop.shape[:2] + (3,))
    new_rgb = hsv2rgb(new_hsv)
  
    # Apply the alpha channel of the source image to the new RGB image
    new_rgb = (1 - source_alpha[..., None]) * backdrop_rgb + source_alpha[..., None] * new_rgb

    # Ensure the RGB values are within the valid range
    new_rgb = np.clip(new_rgb, 0, 1)

    # Convert RGB back to RGBA and scale to 0-255 range
    new_rgba = np.dstack((new_rgb * 255, backdrop[:, :, 3]))

    return new_rgba.astype(np.uint8)

def saturation(backdrop, source, opacity):   
    return hsv(backdrop, source, opacity, "saturation")

def luminance(backdrop, source, opacity):
    return hsv(backdrop, source, opacity, "luminance")

def hue(backdrop, source, opacity):
    return hsv(backdrop, source, opacity, "hue")

def color(backdrop, source, opacity):
    return hsv(backdrop, source, opacity, "color")

def darker_lighter_color(backdrop, source, opacity, type):
    # Normalize the RGB and alpha values to 0-1
    backdrop_norm = backdrop[:, :, :3] / 255
    source_norm = source[:, :, :3] / 255
    source_alpha_norm = source[:, :, 3] / 255

    # Convert RGB to HSV
    backdrop_hsv = np.array([rgb_to_hsv(*rgb) for row in backdrop_norm for rgb in row]).reshape(backdrop.shape[:2] + (3,))
    source_hsv = np.array([rgb_to_hsv(*rgb) for row in source_norm for rgb in row]).reshape(source.shape[:2] + (3,))

    # Create a mask where the value (brightness) of the source image is less than the value of the backdrop image
    if type == "dark":
        mask = source_hsv[:, :, 2] < backdrop_hsv[:, :, 2]
    else:
        mask = source_hsv[:, :, 2] > backdrop_hsv[:, :, 2]

    # Use the mask to select pixels from the source or backdrop
    blend = np.where(mask[..., None], source_norm, backdrop_norm)

    # Apply the alpha channel of the source image to the blended image
    new_rgb = (1 - source_alpha_norm[..., None] * opacity) * backdrop_norm + source_alpha_norm[..., None] * opacity * blend

    # Ensure the RGB values are within the valid range
    new_rgb = np.clip(new_rgb, 0, 1)

    # Convert the RGB values back to 0-255
    new_rgb = new_rgb * 255

    # Calculate the new alpha value by taking the maximum of the backdrop and source alpha channels
    new_alpha = np.maximum(backdrop[:, :, 3], source[:, :, 3])

    # Create a new RGBA image with the calculated RGB and alpha values
    result = np.dstack((new_rgb, new_alpha))

    return result

def darker_color(backdrop, source, opacity):
    return darker_lighter_color(backdrop, source, opacity, "dark")

def lighter_color(backdrop, source, opacity):
    return darker_lighter_color(backdrop, source, opacity, "light")

def simple_mode(backdrop, source, opacity, mode):
    # Normalize the RGB and alpha values to 0-1
    backdrop_norm = backdrop[:, :, :3] / 255
    source_norm = source[:, :, :3] / 255
    source_alpha_norm = source[:, :, 3:4] / 255

    # Calculate the blend without any transparency considerations
    if mode == "linear_burn":
        blend = backdrop_norm + source_norm - 1   
    elif mode == "linear_light":
        blend = backdrop_norm + (2 * source_norm) - 1
    elif mode == "color_dodge":
        blend = backdrop_norm / (1 - source_norm)  
        blend = np.clip(blend, 0, 1) 
    elif mode == "color_burn":
        blend = 1 - ((1 - backdrop_norm) / source_norm)  
        blend = np.clip(blend, 0, 1)   
    elif mode == "exclusion":
        blend = backdrop_norm + source_norm - (2 * backdrop_norm * source_norm)
    elif mode == "subtract":
        blend = backdrop_norm - source_norm
    elif mode == "vivid_light":
        blend = np.where(source_norm <= 0.5, backdrop_norm / (1 - 2 * source_norm), 1 - (1 -backdrop_norm) / (2 * source_norm - 0.5) )
        blend = np.clip(blend, 0, 1)   
    elif mode == "pin_light":
        blend = np.where(source_norm <= 0.5, np.minimum(backdrop_norm, 2 * source_norm), np.maximum(backdrop_norm, 2 * (source_norm - 0.5)))  
    elif mode == "hard_mix":
        blend = simple_mode(backdrop, source, opacity, "linear_light")
        blend = np.round(blend[:, :, :3] / 255)

    # Apply the blended layer back onto the backdrop layer while utilizing the alpha channel and opacity information
    new_rgb = (1 - source_alpha_norm * opacity) * backdrop_norm + source_alpha_norm * opacity * blend

    # Ensure the RGB values are within the valid range
    new_rgb = np.clip(new_rgb, 0, 1)

    # Convert the RGB values back to 0-255
    new_rgb = new_rgb * 255

    # Calculate the new alpha value by taking the maximum of the backdrop and source alpha channels
    new_alpha = np.maximum(backdrop[:, :, 3], source[:, :, 3])

    # Create a new RGBA image with the calculated RGB and alpha values
    result = np.dstack((new_rgb, new_alpha))

    return result

def linear_light(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "linear_light")
def vivid_light(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "vivid_light")
def pin_light(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "pin_light")
def hard_mix(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "hard_mix")
def linear_burn(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "linear_burn")
def color_dodge(backdrop, source, opacity): 
    return simple_mode(backdrop, source, opacity, "color_dodge") 
def color_burn(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "color_burn")
def exclusion(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "exclusion")
def subtract(backdrop, source, opacity):
    return simple_mode(backdrop, source, opacity, "subtract")

BLEND_MODES = {
    "normal": normal,
    "dissolve": dissolve,
    "darken": darken_only,
    "multiply": multiply,
    "color burn": color_burn,
    "linear burn": linear_burn,
    "darker color": darker_color,
    "lighten": lighten_only,
    "screen": screen,
    "color dodge": color_dodge,
    "linear dodge(add)": addition,
    "lighter color": lighter_color,
    "dodge": dodge,
    "overlay": overlay,
    "soft light": soft_light,
    "hard light": hard_light,
    "vivid light": vivid_light,
    "linear light": linear_light,
    "pin light": pin_light,
    "hard mix": hard_mix,
    "difference": difference, 
    "exclusion": exclusion,
    "subtract": subtract,
    "divide": divide,
    "hue": hue,
    "saturation": saturation,
    "color": color,
    "luminosity": luminance,
    "grain extract": grain_extract,
    "grain merge": grain_merge
}
