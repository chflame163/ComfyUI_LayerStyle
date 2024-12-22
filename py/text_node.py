import os
import json
import random
from .imagefunc import AnyType, log, extract_all_numbers_from_str, extract_numbers, extract_substr_from_str
from .imagefunc import tokenize_string, find_best_match_by_similarity, remove_empty_lines, remove_duplicate_string
from .imagefunc import get_files, file_is_extension, is_contain_chinese

any = AnyType("*")

class LS_TextPreseter:
    def __init__(self):
        self.NODE_NAME = "TextPreseter"
    @classmethod
    def INPUT_TYPES(self):
        return {
                    "required":
                    {
                        "title": ("STRING", {"default": "", "multiline": False}),
                        "content": ("STRING", {"default": '', "multiline": True}),                        
                    },
                    "optional": {
                        "text_preset": ("LS_TEXT_PRESET", ),
                    }
                }

    RETURN_TYPES = ("LS_TEXT_PRESET",)
    RETURN_NAMES = ("text_preset",)
    FUNCTION = 'text_preseter'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_preseter(self, title, content, text_preset=None):

        if text_preset is None:
            text_preset = {}

        if title:
            text_preset[title] = content

        return (text_preset,)

class LS_ChoiceTextPreset:
    def __init__(self):
        self.NODE_NAME = "ChoicePresetText"
    @classmethod
    def INPUT_TYPES(self):
        return {
                    "required":
                    {   "text_preset": ("LS_TEXT_PRESET", ),
                        "choice_title": ("STRING", {"default": '', "multiline": False}),
                        "random_choice": ("BOOLEAN", {"default": False}),
                        "default": ("INT", {"default": 0, "min": 0, "max": 1e4, "step": 1}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 1e18, "step": 1}),                        
                    },
                    "optional": {
                    }
                }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("title", "content",)
    FUNCTION = 'choice_preset_text'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def choice_preset_text(self, text_preset, choice_title, random_choice, default, seed):
        keys = list(text_preset.keys())
        ret_key = keys[default]
        ret_value = ''        

        if choice_title in text_preset and not random_choice:
            ret_key = choice_title
        elif random_choice:
            random.seed(seed)
            ret_key = random.choice(list(text_preset.keys()))

        ret_value = text_preset[ret_key]

        return (ret_key, ret_value)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: ChoiceTextPreset": LS_ChoiceTextPreset,
    "LayerUtility: TextPreseter": LS_TextPreseter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ChoiceTextPreset": "LayerUtility: Choice Text Preset",
    "LayerUtility: TextPreseter": "LayerUtility: Text Preseter",
}