from .imagefunc import *
import google.generativeai as genai

NODE_NAME = 'PromptInference'

class PromptInference:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        api_list = ['gemini-pro-vision']

        return {
            "required": {
                "image": ("IMAGE", ),
                "api": (api_list,),
                "use_default_request": ("BOOLEAN", {"default": True}),
                "custom_request": ("STRING", {"default": ""}),
                "key_word": ("STRING", {"default": ""}),
                "exclude_word": ("STRING", {"default": ""}),
                "token_limit": ("INT", {"default": 80, "min": 2, "max": 1024, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'prompt_inference'
    CATEGORY = '😺dzNodes/LayerUtility'
    OUTPUT_NODE = True

    def prompt_inference(self, image, api, use_default_request, custom_request, key_word, exclude_word, token_limit):
        custom_request = custom_request.strip()
        key_word = key_word.strip()
        exclude_word = exclude_word.strip()
        key_word_list = []
        if len(key_word) > 0:
            key_word_list = list(re.split(r'[，,\s*]', key_word))
            key_word_list = [x for x in key_word_list if x != '']  # 去除空字符
            key_word_list = [f"({i})" for i in key_word_list]

        token_prompt = f"as needed keep it under {token_limit} tokens"
        _image = tensor2pil(image).convert('RGB')
        prompt = ""
        ret_text = ""

        if use_default_request:
            prompt = f"{load_inference_prompt()}"
        if len(custom_request) > 0:
            prompt = f"{prompt}{custom_request}, "
        prompt = f"{prompt}{token_prompt}."

        if api == 'gemini-pro-vision':
            model = genai.GenerativeModel(api)
            genai.configure(api_key=get_api_key('google_api_key'), transport='rest')
            response = model.generate_content([prompt, _image])
            ret_text = response.text
            if len(exclude_word) > 0:
                exclude_word_list = list(re.split(r'[，,\s*]', exclude_word))
                exclude_word_list = [x for x in exclude_word_list if x != '']  # 去除空字符
                print(f"exclude_words={exclude_word_list}")
                if len(key_word_list) > 0:
                    ret_text = replace_case(exclude_word_list[0], key_word_list[0], ret_text)
                for i in exclude_word_list:
                    ret_text = replace_case(i, '', ret_text)
                print(f"after exclude_word ret_text = {ret_text}")
                refine_model = genai.GenerativeModel('gemini-pro')
                response = refine_model.generate_content(f"Please correct the grammar errors in the following text:{ret_text}")
                ret_text = response.text

            if len(key_word) > 0:
                ret_text = f"A photo of {', '.join(key_word_list)}, {replace_case('A photo of ', '', ret_text)}, "

            log(f"{NODE_NAME} request to gemini-pro-vision, prompt=\n\033[1;36m{prompt}\033[m\nresponse=\n\033[1;36m{ret_text}\033[m")

        log(f"{NODE_NAME} Processed.", message_type='finish')
        return (ret_text,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: PromptInference": PromptInference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PromptInference": "LayerUtility: PromptInference"
}