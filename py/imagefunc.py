import numpy as np
import torch
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
    ret_mask = torch.tensor([pil2tensor(_image)[0, :, :, 3].tolist()])
    return ret_mask

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
            if x > -distance_x and y > -distance_y:
                if x + distance_x < width and y + distance_y < height:
                    # print(f"x={x}, y={y}, distance_x={distance_x}, distance_y={distance_y}")
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

    ret_mask = torch.cat(out, dim=0)
    # ret_mask = torch.tensor([ret_mask.tolist()])
    return ret_mask
