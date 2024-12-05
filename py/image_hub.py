import torch
import random
from .imagefunc import log



class ImageHub:

    def __init__(self):
        self.NODE_NAME = 'ImageHub'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "output":  ("INT", {"default": 1, "min": 1, "max": 9, "step": 1}),
                "random_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input1_image": ("IMAGE",),
                "input1_mask": ("MASK",),
                "input2_image": ("IMAGE",),
                "input2_mask": ("MASK",),
                "input3_image": ("IMAGE",),
                "input3_mask": ("MASK",),
                "input4_image": ("IMAGE",),
                "input4_mask": ("MASK",),
                "input5_image": ("IMAGE",),
                "input5_mask": ("MASK",),
                "input6_image": ("IMAGE",),
                "input6_mask": ("MASK",),
                "input7_image": ("IMAGE",),
                "input7_mask": ("MASK",),
                "input8_image": ("IMAGE",),
                "input8_mask": ("MASK",),
                "input9_image": ("IMAGE",),
                "input9_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_hub'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_hub(self, output, random_output,
                  input1_image=None, input1_mask=None,
                  input2_image=None, input2_mask=None,
                  input3_image=None, input3_mask=None,
                  input4_image=None, input4_mask=None,
                  input5_image=None, input5_mask=None,
                  input6_image=None, input6_mask=None,
                  input7_image=None, input7_mask=None,
                  input8_image=None, input8_mask=None,
                  input9_image=None, input9_mask=None,
                  ):

        output_list = []
        if input1_image is not None or input1_mask is not None:
            output_list.append(1)
        if input2_image is not None or input2_mask is not None:
            output_list.append(2)
        if input3_image is not None or input3_mask is not None:
            output_list.append(3)
        if input4_image is not None or input4_mask is not None:
            output_list.append(4)
        if input5_image is not None or input5_mask is not None:
            output_list.append(5)
        if input6_image is not None or input6_mask is not None:
            output_list.append(6)
        if input7_image is not None or input7_mask is not None:
            output_list.append(7)
        if input8_image is not None or input8_mask is not None:
            output_list.append(8)
        if input9_image is not None or input9_mask is not None:
            output_list.append(9)

        log(f"output_list={output_list}")
        if len(output_list) == 0:
            log(f"{self.NODE_NAME} is skip, because No Input.", message_type='error')
            return (None, None)

        if random_output:
            index = random.randint(1, len(output_list))
            output = output_list[index - 1]

        ret_image = None
        ret_mask = None
        if output == 1:
            if input1_image is not None:
                ret_image = input1_image
            if input1_mask is not None:
                ret_mask = input1_mask
        elif output == 2:
            if input2_image is not None:
                ret_image = input2_image
            if input2_mask is not None:
                ret_mask = input2_mask
        elif output == 3:
            if input3_image is not None:
                ret_image = input3_image
            if input3_mask is not None:
                ret_mask = input3_mask
        elif output == 4:
            if input4_image is not None:
                ret_image = input4_image
            if input4_mask is not None:
                ret_mask = input4_mask
        elif output == 5:
            if input5_image is not None:
                ret_image = input5_image
            if input5_mask is not None:
                ret_mask = input5_mask
        elif output == 6:
            if input6_image is not None:
                ret_image = input6_image
            if input6_mask is not None:
                ret_mask = input6_mask
        elif output == 7:
            if input7_image is not None:
                ret_image = input7_image
            if input7_mask is not None:
                ret_mask = input7_mask
        elif output == 8:
            if input8_image is not None:
                ret_image = input8_image
            if input8_mask is not None:
                ret_mask = input8_mask
        else:
            if input9_image is not None:
                ret_image = input9_image
            if input9_mask is not None:
                ret_mask = input9_mask

        if ret_image is None and ret_mask is None:
            log(f"{self.NODE_NAME} have {output_list} inputs, output is {output}, but there is no corresponding input.", message_type="error")
        elif ret_image is None:
            log(f"{self.NODE_NAME} have {output_list} inputs, output is {output}, but image is None.", message_type='finish')
        elif ret_mask is None:
            log(f"{self.NODE_NAME} have {output_list} inputs, output is {output}, but mask is None.", message_type='finish')
        else:
            log(f"{self.NODE_NAME} have {output_list} inputs, output is {output}.", message_type='finish')

        return (ret_image, ret_mask)
NODE_CLASS_MAPPINGS = {
    "LayerUtility: ImageHub": ImageHub
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: ImageHub": "LayerUtility: ImageHub"
}