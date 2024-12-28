# layerstyle advance
import os
import torch
import base64
import requests
from io import BytesIO

import folder_paths
from PIL import Image
from .imagefunc import log, tensor2pil, get_api_key


# apikey申请地址：https://bigmodel.cn/usercenter/proj-mgmt/apikeys
class LS_ZhipuImage:

    def __init__(self):
        self.NODE_NAME = 'ZhipuGLM4V'

    @classmethod
    def INPUT_TYPES(cls):
        glm_model_list = ["glm-4v-flash", "glm-4v", "glm-4v-plus"]
        return {"required":{
                    "image": ("IMAGE",),
                    "model": (glm_model_list,),
                    "user_prompt": ("STRING", {"default": "describe this image", "multiline": True}),
                },
                "optional": {
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "zhipu_glm4v"
    CATEGORY = '😺dzNodes/LayerUtility'

    def zhipu_glm4v(self, image, model, user_prompt,):
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=get_api_key('zhipu_api_key'))  # APIKey


        img = tensor2pil(image).convert('RGB')

        img_data = BytesIO()
        img.save(img_data, format="JPEG")
        img_url = base64.b64encode(img_data.getvalue()).decode("utf-8")

        messages = [
                {"role": "user",
                 "content": [
                    {"type": "text",
                     "text": user_prompt
                    },
                    {"type": "image_url",
                     "image_url": {"url": img_url}
                    }
                 ]
                }

            ]

        response = client.chat.completions.create(
            model=model,  # 填写需要调用的模型名称
            messages=messages
        )
        ret_message = response.choices[0].message.content
        log(f"{self.NODE_NAME} response is: {ret_message}")

        return (ret_message,)

class LS_ZhipuText:

    def __init__(self):
        self.NODE_NAME = 'ZhipuGLM4'

    @classmethod
    def INPUT_TYPES(cls):
        glm_model_list = ["GLM-4-Flash", "GLM-4-FlashX", "GLM-4-Plus", "GLM-4-Long","GLM-4-Air", "GLM-4-AirX"]
        return {"required":{
                    "model": (glm_model_list,),
                    "user_prompt": ("STRING", {"default": "where is the capital of France?", "multiline": True}),
                    "history_length": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                },
                "optional": {
                    "history": ("GLM4_HISTORY",),
                }
            }

    RETURN_TYPES = ("STRING", "GLM4_HISTORY",)
    RETURN_NAMES = ("text", "history",)
    FUNCTION = "zhipu_glm4"
    CATEGORY = '😺dzNodes/LayerUtility'

    def zhipu_glm4(self, model, user_prompt, history_length, history=None):

        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=get_api_key('zhipu_api_key'))  # APIKey

        if history is not None:
            messages = history["messages"]
            messages = messages[-history_length *2:]
        else:
            messages = []
        task = {"role": "user", "content": user_prompt}
        messages.append(task)

        response = client.chat.completions.create(
            model=model,  # 填写需要调用的模型名称
            messages=messages
        )
        ret_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ret_message})
        log(f"{self.NODE_NAME} response is: {ret_message}")

        return (ret_message, {"messages":messages},)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: ZhipuGLM4V": LS_ZhipuImage,
    "LayerUtility: ZhipuGLM4": LS_ZhipuText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ZhipuGLM4V": "LayerUtility: ZhipuGLM4V(Advance)",
    "LayerUtility: ZhipuGLM4": "LayerUtility: ZhipuGLM4(Advance)",
}

