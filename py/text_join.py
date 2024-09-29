

NODE_NAME = 'TextJoin'

class TextJoin:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": False,"forceInput":True}),

            },
            "optional": {
                "text_2": ("STRING", {"multiline": False,"forceInput":True}),
                "text_3": ("STRING", {"multiline": False,"forceInput":True}),
                "text_4": ("STRING", {"multiline": False,"forceInput":True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_join(self, **kwargs):

        texts = [kwargs[key] for key in kwargs if key.startswith('text')]
        combined_text = ', '.join(texts)
        return (combined_text.encode('unicode-escape').decode('unicode-escape'),)


class LS_TextJoinV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": False,"forceInput":True}),
                "delimiter": ("STRING", {"default": ",", "multiline": False}),
            },
            "optional": {
                "text_2": ("STRING", {"multiline": False,"forceInput":True}),
                "text_3": ("STRING", {"multiline": False,"forceInput":True}),
                "text_4": ("STRING", {"multiline": False,"forceInput":True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_join(self, text_1, delimiter, text_2=None, text_3=None, text_4=None):

        texts = [text_1]
        if text_2 is not None:
            texts.append(text_2)
        if text_3 is not None:
            texts.append(text_3)
        if text_4 is not None:
            texts.append(text_4)
        combined_text = delimiter.join(texts)

        return (combined_text.encode('unicode-escape').decode('unicode-escape'),)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: TextJoin": TextJoin,
    "LayerUtility: TextJoinV2": LS_TextJoinV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: TextJoin": "LayerUtility: TextJoin",
    "LayerUtility: TextJoinV2": "LayerUtility: TextJoinV2"
}