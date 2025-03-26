import torch
import numpy as np
from PIL import Image, ImageDraw
import math
import random
from .imagefunc import log, tensor2pil, pil2tensor, mask2image


def create_dot_mask(size:int, shape:str='circle') -> np.ndarray:
    """创建不同形状的点阵掩码

    Args:
        size (int): 掩码大小
        shape (str): 形状类型 ('circle', 'diamond', 'square')

    Returns:
        numpy.ndarray: 掩码数组
    """
    mask = np.zeros((size, size))
    center = size / 2

    for x in range(size):
        for y in range(size):
            if shape == 'circle':
                distance = math.sqrt((x - center + 0.5) ** 2 + (y - center + 0.5) ** 2)
                radius = center if size > 4 else center * 1.1
                mask[y, x] = 1 if distance <= radius else 0
            elif shape == 'diamond':
                distance = abs(x - center + 0.5) + abs(y - center + 0.5)
                radius = center if size > 4 else center * 1.1
                mask[y, x] = 1 if distance <= radius else 0
            elif shape == 'square':
                mask[y, x] = 1

    return mask


def halftone(image: Image, dot_size:int = 10, shape: str = 'circle', angle: float = 45) -> Image:

    if image.mode != 'L':
        image = image.convert('L')

    width, height = image.size
    output = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(output)

    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    img_array = np.array(image)

    random_offset = dot_size * 0.05  # 添加 5% 的随机偏移，避免出现规则条纹

    diagonal = math.sqrt(width ** 2 + height ** 2)
    margin = int(diagonal)

    x_start = -margin // 2
    x_end = width + margin // 2
    y_start = -margin // 2
    y_end = height + margin // 2

    step = dot_size
    rotated_step_x = math.sqrt(2) * step * cos_angle
    rotated_step_y = math.sqrt(2) * step * sin_angle

    y = y_start
    while y < y_end:
        x = x_start
        while x < x_end:
            offset_x = random.uniform(-random_offset, random_offset)
            offset_y = random.uniform(-random_offset, random_offset)

            grid_x = (x + offset_x) * cos_angle + (y + offset_y) * sin_angle
            grid_y = -(x + offset_x) * sin_angle + (y + offset_y) * cos_angle

            if 0 <= grid_x < width and 0 <= grid_y < height:
                sample_x = int(grid_x)
                sample_y = int(grid_y)

                region_x = min(sample_x, width - dot_size)
                region_y = min(sample_y, height - dot_size)
                region = img_array[region_y:region_y + dot_size, region_x:region_x + dot_size]

                if region.size > 0:
                    gaussian_kernel = np.exp(-np.linspace(-2, 2, dot_size) ** 2 / 2)
                    gaussian_kernel = gaussian_kernel[:, np.newaxis] * gaussian_kernel[np.newaxis, :]
                    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

                    if region.shape[0] == gaussian_kernel.shape[0] and region.shape[1] == gaussian_kernel.shape[1]:
                        mean_value = np.sum(region * gaussian_kernel)
                    else:
                        mean_value = np.mean(region)

                    dot_radius = math.sqrt(1 - mean_value / 255) * dot_size / 2

                    if dot_radius > 0:
                        mask_size = int(dot_radius * 2)
                        if mask_size > 0:
                            dot_mask = create_dot_mask(mask_size, shape)

                            for dy in range(mask_size):
                                for dx in range(mask_size):
                                    if dot_mask[dy, dx] > 0:
                                        px = int(grid_x - mask_size // 2 + dx)
                                        py = int(grid_y - mask_size // 2 + dy)
                                        if 0 <= px < width and 0 <= py < height:
                                            output.putpixel((px, py), 255)

            x += step
        y += step

    return output


class LS_HalfTone:

    def __init__(self):
        self.NODE_NAME = 'HalfTone'

    @classmethod
    def INPUT_TYPES(self):
        shape_list = ['circle', 'diamond', 'square']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "dot_size": ("INT", {"default": 10, "min": 4, "max": 100, "step": 1}),  # 点大小
                "angle": ("FLOAT", {"default": 45, "min": -90, "max": 90, "step": 0.1}),  # 角度
                "shape": (shape_list,),
                "dot_color":("STRING",{"default": "#000000"}),
                "background_color": ("STRING", {"default": "#FFFFFF"}),
                "anti_aliasing": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'halftone'
    CATEGORY = '😺dzNodes/LayerFilter'

    def halftone(self, image, dot_size, angle, shape, dot_color, background_color, anti_aliasing, mask=None,
                  ):

        l_masks = []
        ret_images = []
        upscale = anti_aliasing + 1
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        else:
            l_masks.append(Image.new('L', tensor2pil(image[0]).size, color='white'))

        for idx,img in enumerate(image):
            orig_image = tensor2pil(img.unsqueeze(0)).convert('RGB')
            orig_mask = l_masks[idx] if len(l_masks) > idx else l_masks[-1]
            if orig_mask.size != orig_image.size:
                orig_mask = orig_mask.resize(orig_image.size, Image.LANCZOS)

            upscaled_image = orig_image.resize((orig_image.width * upscale, orig_image.height * upscale), Image.LANCZOS)

            halftone_image = halftone(upscaled_image, dot_size * upscale, shape=shape, angle=angle)
            halftone_image = halftone_image.resize(orig_image.size, Image.LANCZOS)
            color_image = Image.new('RGB', halftone_image.size, color=dot_color)
            background_image = Image.new('RGB', halftone_image.size, color=background_color)
            background_image.paste(color_image, mask=halftone_image)
            ret_image = Image.new('RGB', halftone_image.size, color=background_color)
            ret_image.paste(background_image, mask=orig_mask)

            ret_images.append(pil2tensor(ret_image))


        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: HalfTone": LS_HalfTone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: HalfTone": "LayerFilter: HalfTone"
}