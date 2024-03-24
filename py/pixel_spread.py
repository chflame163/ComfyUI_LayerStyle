from .imagefunc import *

NODE_NAME = 'PixelSpread'

class PixelSpread:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask
                "mask_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'pixel_spread'
    CATEGORY = '😺dzNodes/LayerMask'

    def pixel_spread(self, image, invert_mask, mask_grow, mask=None):

        l_images = []
        l_masks = []
        ret_images = []

        for l in image:
            i = tensor2pil(torch.unsqueeze(l, 0))
            l_images.append(i)
            if i.mode == 'RGBA':
                l_masks.append(i.split()[-1])
            else:
                l_masks.append(Image.new('L', i.size, 'white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(l_images), len(l_masks))

        for i in range(max_batch):
            _image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            if mask_grow != 0:
                _mask = expand_mask(image2mask(_mask), mask_grow, 0)  # 扩张，模糊
                _mask = mask2image(_mask)
            # i1 = pil2tensor(_image.convert('RGB'))
            # _mask = _mask.convert('RGB')
            if _image.size != _mask.size:
                log(f"Error: {NODE_NAME} skipped, because the mask is not match image.", message_type='error')
                return (image,)
            ret_image = pixel_spread(_image.convert('RGB'), _mask.convert('RGB'))
            ret_images.append(pil2tensor(ret_image))
            #
            # i_dup = copy.deepcopy(i1.cpu().numpy().astype(np.float64))
            # a_dup = copy.deepcopy(pil2tensor(_mask).cpu().numpy().astype(np.float64))
            # fg = copy.deepcopy(i1.cpu().numpy().astype(np.float64))
            #
            # for index, img in enumerate(i_dup):
            #     alpha = a_dup[index][:, :, 0]
            #     fg[index], _ = estimate_foreground_ml(img, np.array(alpha), return_background=True)
            #
            # ret_images.append(torch.from_numpy(fg.astype(np.float32)))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: PixelSpread": PixelSpread
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: PixelSpread": "LayerMask: PixelSpread"
}