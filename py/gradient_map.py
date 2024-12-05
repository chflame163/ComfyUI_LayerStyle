import torch
from PIL import Image
import numpy as np
from .imagefunc import log, tensor2pil, pil2tensor, gradient, Hex_to_RGB



class GradientMap:
    def __init__(self):
        self.NODE_NAME = 'GradientMap'
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "start_color": ("STRING", {"default": "#015A52"}),
                "mid_color": ("STRING", {"default": "#02AF9F"}),
                "end_color": ("STRING", {"default": "#7FFFEC"}),
                "mid_point": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "gradient")
    FUNCTION = 'apply_gradient_map'
    CATEGORY = '😺dzNodes/LayerStyle'

    def apply_gradient_map(self, image, start_color, mid_color, end_color, mid_point, opacity, layer_mask=None):
        def create_gradient_array(start_color, mid_color, end_color, mid_point):
            start_rgb = Hex_to_RGB(start_color)
            mid_rgb = Hex_to_RGB(mid_color)
            end_rgb = Hex_to_RGB(end_color)
            
            mid_index = int(255 * mid_point)
            gradient1 = np.array([np.linspace(start_rgb[i], mid_rgb[i], mid_index + 1) for i in range(3)]).T
            gradient2 = np.array([np.linspace(mid_rgb[i], end_rgb[i], 256 - mid_index) for i in range(3)]).T
            return np.vstack((gradient1[:-1], gradient2))

        gradient_array = create_gradient_array(start_color, mid_color, end_color, mid_point)

        gradient_image = Image.fromarray(np.uint8(gradient_array.reshape(1, -1, 3).repeat(50, axis=0)))
        gradient_tensor = pil2tensor(gradient_image)
        ret_images = []
        for img in image:
            pil_image = tensor2pil(img)
            
            # Convert to grayscale to get luminance
            gray_image = np.array(pil_image.convert('L'))
            
            # Apply gradient map
            gradient_mapped = gradient_array[gray_image]
            
            # Preserve luminance of original image
            original_array = np.array(pil_image)
            luminance = np.sum(original_array * [0.299, 0.587, 0.114], axis=2, keepdims=True) / 255.0
            gradient_mapped = gradient_mapped * luminance + original_array * (1 - luminance)
            
            gradient_mapped_image = Image.fromarray(np.uint8(gradient_mapped))
            
            # Apply opacity
            if opacity < 100:
                gradient_mapped_image = Image.blend(pil_image, gradient_mapped_image, opacity / 100)
            
            # Apply mask if provided
            if layer_mask is not None:
                mask = tensor2pil(layer_mask).convert('L')
                pil_image.paste(gradient_mapped_image, (0, 0), mask)
            else:
                pil_image = gradient_mapped_image
            
            ret_images.append(pil2tensor(pil_image))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), gradient_tensor)
    
NODE_CLASS_MAPPINGS = {
    "LayerStyle: Gradient Map": GradientMap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerStyle: Gradient Map": "LayerStyle: Gradient Map"
}