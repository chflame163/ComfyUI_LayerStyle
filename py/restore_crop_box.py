import torch
from PIL import Image
from .imagefunc import log, tensor2pil, pil2tensor, image2mask



class RestoreCropBox:

    def __init__(self):
        self.NODE_NAME = 'RestoreCropBox'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "crop_box": ("BOX",),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = '😺dzNodes/LayerUtility'

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box,
                         croped_mask=None
                         ):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in croped_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', size=m.size, color='white'))
        if croped_mask is not None:
            if croped_mask.dim() == 2:
                croped_mask = torch.unsqueeze(croped_mask, 0)
            l_masks = []
            for m in croped_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            croped_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(croped_image).convert('RGB')

            ret_mask = Image.new('L', size=_canvas.size, color='black')
            _canvas.paste(_layer, box=tuple(crop_box), mask=_mask)
            ret_mask.paste(_mask, box=tuple(crop_box))
            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(ret_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: RestoreCropBox": RestoreCropBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RestoreCropBox": "LayerUtility: RestoreCropBox"
}