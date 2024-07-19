import os.path
from pathlib import Path
import torch
from PIL import Image
import math
from torchvision.transforms import ToPILImage
import folder_paths
from .imagefunc import files_for_uform_gen2_qwen, StopOnTokens, UformGen2QwenChat, clear_memory

# Example of integrating UformGen2QwenChat into a node-like structure
class QWenImage2Prompt:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": False, "default": "describe this image",},),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "uform_gen2_qwen_chat"
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def uform_gen2_qwen_chat(self, image, question):
        chat_model = UformGen2QwenChat()
        history = []  # Example empty history
        pil_image = ToPILImage()(image[0].permute(2, 0, 1))
        width, height = pil_image.size
        ratio = width / height
        if width * height > 1024 * 1024:
            target_width = math.sqrt(ratio * 1024 * 1024)
            target_height = target_width / ratio
            target_width = int(target_width)
            target_height = int(target_height)
            pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        temp_path = files_for_uform_gen2_qwen / "temp.png"
        pil_image.save(temp_path)
        question = f"{question} but output no more then 80 words."
        response = chat_model.chat_response(question, history, temp_path)

        # Cleanup
        del chat_model
        clear_memory()
        return (response.split("assistant\n", 1)[1], )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: QWenImage2Prompt": QWenImage2Prompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: QWenImage2Prompt": "LayerUtility: QWenImage2Prompt"
}