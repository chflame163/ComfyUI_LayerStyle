

NODE_NAME = 'TextJoin'

class TextJoin:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
            },
            "optional": {
                "text_3": ("STRING", {"multiline": True}),
                "text_4": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '😺dzNodes/LayerUtility'

    def text_join(self, **kwargs):

        texts = [kwargs[key] for key in kwargs if key.startswith('text')]
        combined_text = ', '.join(texts)
        return (combined_text,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: TextJoin": TextJoin
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: TextJoin": "LayerUtility: TextJoin"
}