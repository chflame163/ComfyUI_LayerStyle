import copy
from pymatting import *
from .imagefunc import *

class PixelSpread:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'pixel_spread'
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def pixel_spread(self, image, invert_mask, mask=None):
        _image = tensor2pil(image)
        if _image.mode == 'RGBA':
            _mask = _image.split()[-1]
        else:
            _mask = Image.new('L', _image.size, 'white')
        if mask is not None:
            if invert_mask:
                mask = 1 - mask
            _mask = mask2image(mask).convert('L')
        image = pil2tensor(_image.convert('RGB'))
        _mask = _mask.convert('RGB')
        i_dup = copy.deepcopy(image.cpu().numpy().astype(np.float64))
        a_dup = copy.deepcopy(pil2tensor(_mask).cpu().numpy().astype(np.float64))
        fg = copy.deepcopy(image.cpu().numpy().astype(np.float64))

        for index, image in enumerate(i_dup):
            trimap = a_dup[index][:, :, 0]  # convert to single channel
            trimap = fix_trimap(trimap, 0.01, 0.99)
            alpha = estimate_alpha_cf(image, trimap, laplacian_kwargs={"epsilon": 1e-6},
                                      cg_kwargs={"maxiter": 100})
            fg[index], _ = estimate_foreground_ml(image, np.array(alpha), return_background=True)

        return (torch.from_numpy(fg.astype(np.float32)),  # fg
                )

NODE_CLASS_MAPPINGS = {
    "LayerMask: PixelSpread": PixelSpread
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: PixelSpread": "LayerMask: PixelSpread"
}