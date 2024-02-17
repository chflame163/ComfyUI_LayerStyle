from .imagefunc import *

NODE_NAME = 'CreateGradientMask'

any = AnyType("*")
class CreateGradientMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        side = ['bottom', 'top', 'left', 'right']
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "gradient_side": (side,),
                "gradient_scale": ("INT", {"default": 100, "min": 1, "max": 9999, "step": 1}),
                "gradient_offset": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = 'create_gradient_mask'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def create_gradient_mask(self, width, height, gradient_side, gradient_scale, gradient_offset, opacity, size_as=None):

        if size_as is not None:
            if size_as.shape[0] > 0:
                _asimage = tensor2pil(size_as[0])
            else:
                _asimage = tensor2pil(size_as)
            width, height = _asimage.size

        _black = Image.new('L', size=(width, height), color='black')
        _white = Image.new('L', size=(width, height), color='white')
        _canvas = copy.deepcopy(_black)
        start_color = '#FFFFFF'
        end_color = '#000000'
        if gradient_side == 'bottom':
            _gradient = create_gradient(start_color, end_color, width, height, direction='bottom')
            if gradient_scale != 100:
                _gradient = _gradient.resize((width, int(height * gradient_scale / 100)))
            _canvas.paste(_gradient.convert('L'), box=(0, gradient_offset))
            if gradient_offset > height:
                _canvas = _white
            elif gradient_offset > 0:
                _canvas.paste(_white, box=(0, gradient_offset - height))
        elif gradient_side == 'top':
            _gradient = create_gradient(start_color, end_color, width, height, direction='top')
            if gradient_scale != 100:
                _gradient = _gradient.resize((width, int(height * gradient_scale / 100)))
            _canvas.paste(_gradient.convert('L'), box=(0, height - int(height * gradient_scale / 100) + gradient_offset))
            if gradient_offset < -height:
                _canvas = _white
            elif gradient_offset < 0:
                _canvas.paste(_white, box=(0, height + gradient_offset))
        elif gradient_side == 'left':
            _gradient = create_gradient(start_color, end_color, width, height, direction='left')
            if gradient_scale != 100:
                _gradient = _gradient.resize((int(width * gradient_scale / 100), height))
            _canvas.paste(_gradient.convert('L'), box=(width - int(width * gradient_scale / 100) + gradient_offset, 0))
            if gradient_offset < -width:
                _canvas = _white
            elif gradient_offset < 0:
                _canvas.paste(_white, box=(width + gradient_offset, 0))
        elif gradient_side == 'right':
            _gradient = create_gradient(start_color, end_color, width, height, direction='right')
            if gradient_scale != 100:
                _gradient = _gradient.resize((int(width * gradient_scale / 100), height))
            _canvas.paste(_gradient.convert('L'), box=(gradient_offset, 0))
            if gradient_offset > width:
                _canvas = _white
            elif gradient_offset > 0:
                _canvas.paste(_white, box=(gradient_offset - width, 0))

        # opacity
        if opacity < 100:
            _canvas = chop_image(_black, _canvas, 'normal', opacity)

        log(f"{NODE_NAME} Processed.", message_type='finish')
        return (image2mask(_canvas),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: CreateGradientMask": CreateGradientMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: CreateGradientMask": "LayerMask: CreateGradientMask"
}