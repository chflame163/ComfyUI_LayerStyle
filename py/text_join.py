


class TextJoin:

    def __init__(self):
        self.NODE_NAME = 'TextJoin'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"default": "", "multiline": False,"forceInput":False}),

            },
            "optional": {
                "text_2": ("STRING", {"default": "", "multiline": False,"forceInput":False}),
                "text_3": ("STRING", {"default": "", "multiline": False,"forceInput":False}),
                "text_4": ("STRING", {"default": "", "multiline": False,"forceInput":False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_join(self, text_1, text_2="", text_3="", text_4=""):

        texts = []
        if text_1 != "":
            texts.append(text_1)
        if text_2 != "":
            texts.append(text_2)
        if text_3 != "":
            texts.append(text_3)
        if text_4 != "":
            texts.append(text_4)
        if len(texts) > 0:
            combined_text = ', '.join(texts)
            return (combined_text.encode('unicode-escape').decode('unicode-escape'),)
        else:
            return ('',)


class LS_TextJoinV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"default": "", "multiline": False,"forceInput":True}),
                "delimiter": ("STRING", {"default": ",", "multiline": False}),
            },
            "optional": {
                "text_2": ("STRING", {"default": "", "multiline": False,"forceInput":True}),
                "text_3": ("STRING", {"default": "", "multiline": False,"forceInput":True}),
                "text_4": ("STRING", {"default": "", "multiline": False,"forceInput":True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_join(self, text_1, delimiter, text_2="", text_3="", text_4=""):

        texts = []
        if text_1 != "":
            texts.append(text_1)
        if text_2 != "":
            texts.append(text_2)
        if text_3 != "":
            texts.append(text_3)
        if text_4 != "":
            texts.append(text_4)
        if len(texts) > 0:
            combined_text = delimiter.join(texts)
            return (combined_text.encode('unicode-escape').decode('unicode-escape'),)
        else:
            return ('',)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: TextJoin": TextJoin,
    "LayerUtility: TextJoinV2": LS_TextJoinV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: TextJoin": "LayerUtility: TextJoin",
    "LayerUtility: TextJoinV2": "LayerUtility: TextJoinV2"
}