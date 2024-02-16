from .imagefunc import *

NODE_NAME = 'Exposure'

class ColorCorrectExposure:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "exposure": ("INT", {"default": 20, "min": -100, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_exposure'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_correct_exposure(self, image, exposure):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            t = i.detach().clone().cpu().numpy().astype(np.float32)
            more = t[:, :, :, :3] > 0
            t[:, :, :, :3][more] *= pow(2, exposure / 32)
            if exposure < 0:
                bp = -exposure / 250
                scale = 1 / (1 - bp)
                t = np.clip((t - bp) * scale, 0.0, 1.0)
            ret_images.append(torch.from_numpy(t))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "LayerColor: Exposure": ColorCorrectExposure
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: Exposure": "LayerColor: Exposure"
}