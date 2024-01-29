'''Image process functions for ComfyUI nodes
by chflame https://github.com/chflame163
'''
import numpy as np
import torch
import scipy.ndimage
import cv2
from typing import Union, List
from PIL import Image, ImageFilter, ImageChops, ImageDraw
import colorsys

def log(message):
    name = 'LayerStyle'
    print(f"# 😺dzNodes: {name} -> {message}")

'''Converter'''

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

'''Image Functions'''

def shift_image(image:Image, distance_x:int, distance_y:int) -> Image:
    bkcolor = (0, 0, 0)
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=bkcolor)
    for x in range(width):
        for y in range(height):
            if x > -distance_x and y > -distance_y:  # 防止回转
                if x + distance_x < width and y + distance_y < height:  # 防止越界
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

def remove_background(image:Image, mask:Image, color:str) -> Image:
    width = image.width
    height = image.height
    ret_image = Image.new('RGB', size=(width, height), color=color)
    ret_image.paste(image, mask=mask)
    return ret_image

def gaussian_blur(image:Image, radius:int) -> Image:
    image = image.convert("RGBA")
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


def create_gradient(start_color_inhex:str, end_color_inhex:str, width:int, height:int) -> Image:
    start_color = Hex_to_RGB(start_color_inhex)
    end_color = Hex_to_RGB(end_color_inhex)
    ret_image = Image.new("RGB", (width, height), start_color)
    draw = ImageDraw.Draw(ret_image)
    for i in range(height):
        R = int(start_color[0] * (height - i) / height + end_color[0] * i / height)
        G = int(start_color[1] * (height - i) / height + end_color[1] * i / height)
        B = int(start_color[2] * (height - i) / height + end_color[2] * i / height)
        color = (R, G, B)
        draw.line((0, i, width, i), fill=color)
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
    image = image.convert('RGBA')
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + width, y + height), fill=box_color, outline=line_color, width=line_width, )
    return image

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

'''Mask Functions'''

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
    _image = mask2image(mask)
    return image2mask(ImageChops.invert(_image))

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
    log(f"x1={x1}, y1={y1},x2={x2}, y2={y2}")
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


