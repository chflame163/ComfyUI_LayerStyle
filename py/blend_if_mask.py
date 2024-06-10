from .imagefunc import *

NODE_NAME = 'BlendIfMask'

def norm_value(value):
    if value < 0.01:
        value = 0.01
    if value > 0.99:
        value = 0.99
    return value

class BlendIfMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        blend_if_list = ["gray", "red", "green", "blue"]
        return {
            "required": {
                "image": ("IMAGE", ),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_if": (blend_if_list,),
                "black_point": ("INT", {"default": 0, "min": 0, "max": 254, "step": 1, "display": "slider"}),
                "black_range": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "white_point": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1, "display": "slider"}),
                "white_range": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'blend_if_mask'
    CATEGORY = '😺dzNodes/LayerMask'

    def blend_if_mask(self, image, invert_mask, blend_if,
                      black_point, black_range,
                      white_point, white_range,
                      mask=None
                      ):


        ret_masks = []
        input_images = []
        input_masks = []

        for i in image:
            input_images.append(torch.unsqueeze(i, 0))
            m = tensor2pil(i)
            if m.mode == 'RGBA':
                input_masks.append(m.split()[-1])
            else:
                input_masks.append(Image.new('L', size=m.size, color='white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            input_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                input_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(input_images), len(input_masks))

        for i in range(max_batch):
            _image = input_images[i] if i < len(input_images) else input_images[-1]
            _image = tensor2pil(_image).convert('RGB')

            if blend_if == "red":
                gray = image_channel_split(_image, 'RGB')[0]
            elif blend_if == "green":
                gray = image_channel_split(_image, 'RGB')[1]
            elif blend_if == "blue":
                gray = image_channel_split(_image, 'RGB')[2]
            else:
                gray = _image.convert('L')

            _mask = input_masks[i] if i < len(input_masks) else input_masks[-1]

            gray = histogram_range(gray, black_point, black_range, white_point, white_range)
            black = Image.new('L', size=_image.size, color='black')
            _mask = ImageChops.invert(_mask)
            gray.paste(black, mask=_mask)

            ret_masks.append(image2mask(gray))

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: BlendIf Mask": BlendIfMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: BlendIf Mask": "LayerMask: BlendIf Mask"
}