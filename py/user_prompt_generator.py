from .imagefunc import log

class LS_UserPromptGenerator_Txt2ImgPrompt:

    def __init__(self):
        self.NODE_NAME = 'UserPromptGenerator-Txt2ImgPrompt'

    @classmethod
    def INPUT_TYPES(self):
        template_list = ['SD txt2img prompt',]
        return {
            "required": {
                "template": (template_list,),
                "describe": ("STRING", {"default": "1 girl","multiline": True}),
                "limit_words": ("INT", {"default": 200, "min": 2, "max": 999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("user_prompt", )
    FUNCTION = 'user_prompt_generator_txt2img_prompt'
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def user_prompt_generator_txt2img_prompt(self, template, describe, limit_words):

        if template == 'SD txt2img prompt':
            prompt = (f"You are creating a prompt for Stable Diffusion to generate an image.\r\n"
                      f"Using '{describe}' as the basic content, polish and embellish it to describe into text.\r\n"
                      f"The word limit for the answer is between {int(limit_words * 0.7)} - {int(limit_words * 1.1)} words. Not too little, nor too much.\r\n"
                      f"Only output the prompt itself, don't output any unnecessary content like word count info.")

        log(f'{self.NODE_NAME} Processed. result is \r\n"{prompt}".')
        return (prompt,)


class LS_UserPromptGenerator_ReplaceWord:

    def __init__(self):
        self.NODE_NAME = 'UserPromptGenerator-ReplaceWord'

    @classmethod
    def INPUT_TYPES(self):
        template_list = ['prompt replace word', ]
        return {
            "required": {
                "orig_prompt": ("STRING", {"forceInput":True}),
                "template": (template_list,),
                "exclude_word": ("STRING", {"default": ""}),
                "replace_with_word": ("STRING", {"default": ""}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("user_prompt",)
    FUNCTION = 'user_prompt_generator_replace_word'
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def user_prompt_generator_replace_word(self, orig_prompt, template, exclude_word, replace_with_word):
        if template == 'prompt replace word':
            prompt = (f'You are creating a prompt for Stable Diffusion to generate an image. '
                      f'First step: Replace "{exclude_word}" and its synonyms with "{replace_with_word}" in the following text:"{orig_prompt}".\r\n'
                      f'Second step: Correct the grammar errors for based on first step.\r\n'
                      f"Only output the second step result, don't output any unnecessary content like first step result."
                      )

        log(f'{self.NODE_NAME} Processed. result is \r\n"{prompt}".')
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: UserPromptGeneratorTxt2ImgPrompt": LS_UserPromptGenerator_Txt2ImgPrompt,
    "LayerUtility: UserPromptGeneratorReplaceWord": LS_UserPromptGenerator_ReplaceWord
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: UserPromptGeneratorTxt2ImgPrompt": "LayerUtility: UserPrompt Generator Txt2Img",
    "LayerUtility: UserPromptGeneratorReplaceWord": "LayerUtility: UserPrompt Generator Replace Word"
}