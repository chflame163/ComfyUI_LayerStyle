import torch
import copy
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask, chop_image, gradient, mask_area



class MaskGradient:

    def __init__(self):
        self.NODE_NAME = 'MaskGradient'

    @classmethod
    def INPUT_TYPES(self):
        side = ['top', 'bottom', 'left', 'right']
        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "gradient_side": (side,),
                "gradient_scale": ("INT", {"default": 100, "min": 1, "max": 9999, "step": 1}),
                "gradient_offset": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_gradient'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_gradient(self, mask, invert_mask, gradient_side, gradient_scale, gradient_offset, opacity, ):


        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        l_masks = []
        ret_masks = []

        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            _canvas = copy.copy(_mask)
            width = _mask.width
            height = _mask.height
            _gradient = gradient('#000000', '#FFFFFF',
                                 1024, 1024, 0)
            # (box_x, box_y, box_width, box_height) = min_bounding_rect(_mask)
            (box_x, box_y, box_width, box_height) = mask_area(_mask)
            log(f"{self.NODE_NAME}: Box detected. x={box_x},y={box_y},width={box_width},height={box_height}")
            if box_width < 1 or box_height < 1:
                log(f"Error: {self.NODE_NAME} skipped, because the mask is does'nt have valid area", message_type='error')
                return (mask,)

            if gradient_side == 'top':
                boxsize = (width, box_height)
                _gradient = _gradient.transpose(Image.FLIP_TOP_BOTTOM)
                _gradient = _gradient.resize(boxsize)
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((width, int(box_height * gradient_scale / 100)))
                    _box.paste(_gradient, box = (0, 0))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(0, gradient_offset))
                    _box.paste(_boxwhite, box = (0, gradient_offset - _box.height))
                    _gradient = _box
                    if gradient_offset > box_height:
                        _gradient = Image.new('RGB', size = boxsize, color = 'white')
                _canvas.paste(_black, box = (0, box_y), mask = _gradient.convert('L'))
            elif gradient_side == 'bottom':
                boxsize = (width, box_height)
                _gradient = _gradient.resize((width, box_height))
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((width, int(box_height * gradient_scale / 100)))
                    _box.paste(_gradient, box = (0, box_height - _gradient.height))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(0, gradient_offset))
                    _box.paste(_boxwhite, box = (0, gradient_offset + _box.height))
                    _gradient = _box
                    if gradient_offset < -box_height:
                        _gradient = Image.new('RGB', size=boxsize, color='white')
                _canvas.paste(_black, box = (0, box_y + 1), mask = _gradient.convert('L'))
            elif gradient_side == 'left':
                boxsize = (box_width, height)
                _gradient = _gradient.transpose(Image.ROTATE_270)
                _gradient = _gradient.resize(boxsize)
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((int(box_width * gradient_scale / 100), height))
                    _box.paste(_gradient, box = (0, 0))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(gradient_offset, 0))
                    _box.paste(_boxwhite, box = (gradient_offset - _box.width, 0))
                    _gradient = _box
                    if gradient_offset > box_width:
                        _gradient = Image.new('RGB', size=boxsize, color='white')
                _canvas.paste(_black, box = (box_x, 0), mask = _gradient.convert('L'))
            elif gradient_side == 'right':
                boxsize = (box_width, height)
                _gradient = _gradient.transpose(Image.ROTATE_90)
                _gradient = _gradient.resize(boxsize)
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((int(box_width * gradient_scale / 100), height))
                    _box.paste(_gradient, box = (box_width - _gradient.width, 0))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(gradient_offset, 0))
                    _box.paste(_boxwhite, box = (gradient_offset + _box.width, 0))
                    _gradient = _box
                    if gradient_offset < -box_width:
                        _gradient = Image.new('RGB', size=boxsize, color='white')

                _canvas.paste(_black, box = (box_x + 1, 0), mask = _gradient.convert('L'))
            # opacity
            if opacity < 100:
                _canvas = chop_image(_mask, _canvas, 'normal', opacity)
            ret_masks.append(image2mask(_canvas))

        log(f"{self.NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskGradient": MaskGradient
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskGradient": "LayerMask: MaskGradient"
}