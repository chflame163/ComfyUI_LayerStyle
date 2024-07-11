from .imagefunc import *

NODE_NAME = 'PromptEmbellish'

class PromptEmbellish:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        api_list = ['gemini-1.5-flash', 'gemini-pro-vision']
        return {
            "required": {
                "api": (api_list,),
                "token_limit": ("INT", {"default": 40, "min": 2, "max": 1024, "step": 1}),
                "describe": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'prompt_embellish'
    CATEGORY = '😺dzNodes/LayerUtility/Prompt'

    def prompt_embellish(self, api, token_limit, describe, image=None):
        if describe == "" and image is None:
            return ("",)
        import google.generativeai as genai
        ret_text = ""
        first_step_prompt = (f"You are creating a prompt for Stable Diffusion to generate an image. "
                             f"First step:Using '{describe}' as the basic content, "
                             f"polish and embellish it to describe into text, keep it on {token_limit} tokens."
                             f"Second step: Generate a Stable Diffusion text prompt for based on first step in at least {token_limit} words."
                             f"Only respond with the prompt itself, but embellish it."
                             )

        genai.configure(api_key=get_api_key('google_api_key'), transport='rest')
        if describe != "":
            model = genai.GenerativeModel('gemini-pro',
                                          generation_config=gemini_generate_config,
                                          safety_settings=gemini_safety_settings)
            log(f"{NODE_NAME}: Request to gemini-pro...")
            response = model.generate_content(first_step_prompt)
            print(response)
            ret_text = response.text
            ret_text = ret_text[ret_text.rfind(':') + 1:]
            ret_text = ret_text[ret_text.rfind('\n') + 1:]
            # log(f"{NODE_NAME}: Text2Image Prompt is:\n\033[1;36m{ret_text}\033[m")
            if is_contain_chinese(describe):
                translate_prompt = (f"Please translate the text in parentheses into English:({describe})"
                                    )
                response = model.generate_content(translate_prompt)
                print(response)
                ret_discribe = response.text
            else:
                ret_discribe = describe

        if image is not None:
            if describe != "":
                second_step_prompt = (f"You are creating a prompt for Stable Diffusion to generate an image. "
                                      f"First step:Modify and polish the content in parentheses to match this photo,"
                                      f"but must keep '{describe}': ({ret_text}) "
                                      f"Second step: Find objects that is similar in parentheses from the content of the first step"
                                      f" and replace it with the content in parentheses: ({describe})"
                                      f"Third step: Generate a Stable Diffusion text prompt for based on second step in at least {token_limit} words."                                 
                                      f"Only respond with the prompt itself, but embellish it."
                                      )
            else:
                second_step_prompt = (f"You are creating a prompt for Stable Diffusion to generate an image. "
                                      f"First step: describe this image, "
                                      f"polish and embellish it into text, discrete it in {token_limit} tokens."
                                      f"Second step: Generate a Stable Diffusion text prompt for based on first step in at least {token_limit} words."
                                      f"Only respond with the prompt itself, but embellish it."
                                      )
            _image = tensor2pil(image).convert('RGB')
            model = genai.GenerativeModel(api,
                                          generation_config=gemini_generate_config,
                                          safety_settings=gemini_safety_settings)
            log(f"{NODE_NAME}: Request to {api}...")
            response = model.generate_content([second_step_prompt, _image])
            print(response)
            ret_text = response.text
            ret_text = ret_text[ret_text.rfind(':') + 1:]
            ret_text = ret_text.replace('(','').replace(')','')
            if describe != "":
                ret_text = f"((({ret_discribe}))), {ret_text}"
            # log(f"{NODE_NAME}: Text2Image by ImageRefrence Prompt is:\n\033[1;36m{ret_text}\033[m")
        log(f"{NODE_NAME}: Prompt is:\n\033[1;36m{ret_text}\033[m")
        log(f"{NODE_NAME} Processed.", message_type='finish')
        return (ret_text,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: PromptEmbellish": PromptEmbellish
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PromptEmbellish": "LayerUtility: PromptEmbellish"
}