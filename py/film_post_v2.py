from .imagefunc import *

NODE_NAME = 'FilmV2'

class FilmV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        grain_method_list = ["fastgrain", "filmgrainer", ]
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0.01, "max": 3, "step": 0.01}),
                "vignette_intensity": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "grain_method": (grain_method_list,),
                "grain_power": ("FLOAT", {"default": 0.15, "min": 0, "max": 1, "step": 0.01}),
                "grain_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.1}),
                "grain_sat": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "filmgrainer_shadows": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "filmgrainer_highs": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
                "blur_strength": ("INT", {"default": 90, "min": 0, "max": 256, "step": 1}),
                "blur_focus_spread": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 8, "step": 0.1}),
                "focal_depth": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1, "step": 0.01}),
            },
            "optional": {
                "depth_map": ("IMAGE",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'film_v2'
    CATEGORY = '😺dzNodes/LayerFilter'

    def film_v2(self, image, center_x, center_y, saturation, vignette_intensity,
                  grain_method, grain_power, grain_scale, grain_sat, filmgrainer_shadows, filmgrainer_highs,
                  blur_strength, blur_focus_spread, focal_depth,
                  depth_map=None
                  ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')

            if saturation != 1:
                color_image = ImageEnhance.Color(_canvas)
                _canvas = color_image.enhance(factor= saturation)

            if blur_strength:
                if depth_map is not None:
                    depth_map = tensor2pil(depth_map).convert('RGB')
                    if depth_map.size != _canvas.size:
                        depth_map.resize((_canvas.size), Image.BILINEAR)
                    _canvas = depthblur_image(_canvas, depth_map, blur_strength, focal_depth, blur_focus_spread)
                else:
                    _canvas = radialblur_image(_canvas, blur_strength, center_x, center_y, blur_focus_spread * 2)

            if vignette_intensity:
                # adjust image gamma and saturation
                _canvas = gamma_trans(_canvas, 1 - vignette_intensity / 3)
                color_image = ImageEnhance.Color(_canvas)
                _canvas = color_image.enhance(factor= 1+ vignette_intensity / 3)
                # add vignette
                _canvas = vignette_image(_canvas, vignette_intensity, center_x, center_y)

            if grain_power:
                if grain_method == "fastgrain":
                    _canvas = image_add_grain(_canvas, grain_scale,grain_power, grain_sat, toe=0, seed=int(time.time()))
                elif grain_method == "filmgrainer":
                    _canvas = filmgrain_image(_canvas, grain_scale, grain_power, filmgrainer_shadows, filmgrainer_highs, grain_sat)

            ret_image = _canvas
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "LayerFilter: FilmV2": FilmV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerFilter: FilmV2": "LayerFilter: Film V2"
}