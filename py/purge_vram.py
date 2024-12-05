import torch.cuda
import gc
import comfy.model_management
from .imagefunc import AnyType, clear_memory

any = AnyType("*")

class PurgeVRAM:

    def __init__(self):
        self.NODE_NAME = 'PurgeVRAM'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "purge_vram"
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'
    OUTPUT_NODE = True

    def purge_vram(self, anything, purge_cache, purge_models):
        clear_memory()
        if purge_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        return (None,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: PurgeVRAM": PurgeVRAM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PurgeVRAM": "LayerUtility: Purge VRAM"
}