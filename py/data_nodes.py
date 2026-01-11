import torch
from .imagefunc import AnyType, log, extract_all_numbers_from_str


any = AnyType("*")

class SeedNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "seed":("INT", {"default": 0, "min": 0, "max": 1e18, "step": 1}),
            },}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = 'seed_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def seed_node(self, seed):
        return (seed,)

class BooleanOperator:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["==", "!=", ">", "<", ">=", "<=", "and", "or", "xor", "not(a)", "min", "max"]
        return {"required": {
                "a": (any, ),
                "b": (any, ),
                "operator": (operator_list,),
            },}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("output",)
    FUNCTION = 'bool_operator_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def bool_operator_node(self, a, b, operator):
        ret_value = False
        if operator == "==":
            ret_value = a == b
        if operator == "!=":
            ret_value = a != b
        if operator == ">":
            ret_value = a > b
        if operator == "<":
            ret_value = a < b
        if operator == ">=":
            ret_value = a >= b
        if operator == "<=":
            ret_value = a <= b
        if operator == "and":
            ret_value = a and b
        if operator == "or":
            ret_value = a or b
        if operator == "xor":
            ret_value = not(a == b)
        if operator == "not(a)":
            ret_value = not a
        if operator == "min":
            ret_value = min(a, b)
        if operator == "max":
            ret_value = max(a, b)

        return (ret_value,)

class BooleanOperatorV2:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["==", "!=", ">", "<", ">=", "<=", "and", "or", "xor", "not(a)", "min", "max"]
        return {
                    "required":
                    {
                        "a_value": ("STRING", {"default": "", "multiline": False}),
                        "b_value": ("STRING", {"default": "", "multiline": False}),
                        "operator": (operator_list,),
                    },
                    "optional": {
                        "a": (any,),
                        "b": (any,),
                    }
                }

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("output", "string",)
    FUNCTION = 'bool_operator_node_v2'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def bool_operator_node_v2(self, a_value, b_value, operator, a = None, b = None):
        if a is None:
            if a_value != "":
                _numbers = extract_all_numbers_from_str(a_value, checkint=True)
                if len(_numbers) > 0:
                    a = _numbers[0]
                else:
                    a = 0
            else:
                a = 0

        if b is None:
            if b_value != "":
                _numbers = extract_all_numbers_from_str(b_value, checkint=True)
                if len(_numbers) > 0:
                    b = _numbers[0]
                else:
                    b = 0
            else:
                b = 0

        ret_value = False
        if operator == "==":
            ret_value = a == b
        if operator == "!=":
            ret_value = a != b
        if operator == ">":
            ret_value = a > b
        if operator == "<":
            ret_value = a < b
        if operator == ">=":
            ret_value = a >= b
        if operator == "<=":
            ret_value = a <= b
        if operator == "and":
            ret_value = a and b
        if operator == "or":
            ret_value = a or b
        if operator == "xor":
            ret_value = not(a == b)
        if operator == "not(a)":
            ret_value = not a
        if operator == "min":
            ret_value = min(a, b)
        if operator == "max":
            ret_value = max(a, b)

        return (ret_value, str(ret_value))

class NumberCalculator:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["+", "-", "*", "/", "**", "//", "%", "nth_root", "min", "max"]
        return {"required": {
                "a": (any, {}),
                "b": (any, {}),
                "operator": (operator_list,),
            },}

    RETURN_TYPES = ("INT", "FLOAT",)
    RETURN_NAMES = ("int", "float",)
    FUNCTION = 'number_calculator_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def number_calculator_node(self, a, b, operator):
        ret_value = 0
        if operator == "+":
            ret_value = a + b
        if operator == "-":
            ret_value = a - b
        if operator == "*":
            ret_value = a * b
        if operator == "**":
            ret_value = a ** b
        if operator == "%":
            ret_value = a % b
        if operator == "nth_root":
            ret_value = a ** (1/b)
        if operator == "min":
            ret_value = min(a, b)
        if operator == "max":
            ret_value = max(a, b)
        if operator == "/":
            if b != 0:
                ret_value = a / b
            else:
                ret_value = 0
        if operator == "//":
            if b != 0:
                ret_value = a // b
            else:
                ret_value = 0

        return (int(ret_value), float(ret_value),)

class NumberCalculatorV2:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["+", "-", "*", "/", "**", "//", "%" , "nth_root", "min", "max"]

        return {
                    "required":
                    {
                        "a_value": ("STRING", {"default": "", "multiline": False}),
                        "b_value": ("STRING", {"default": "", "multiline": False}),
                        "operator": (operator_list,),
                    },
                    "optional": {
                        "a": (any,),
                        "b": (any,),
                    }
                }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("int", "float", "string",)
    FUNCTION = 'number_calculator_node_v2'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def number_calculator_node_v2(self, a_value, b_value, operator, a = None, b = None):
        if a is None:
            if a_value != "":
                _numbers = extract_all_numbers_from_str(a_value, checkint=True)
                if len(_numbers) > 0:
                    a = _numbers[0]
                else:
                    a = 0
            else:
                a = 0

        if b is None:
            if b_value != "":
                _numbers = extract_all_numbers_from_str(b_value, checkint=True)
                if len(_numbers) > 0:
                    b = _numbers[0]
                else:
                    b = 0
            else:
                b = 0

        ret_value = 0
        if operator == "+":
            ret_value = a + b
        if operator == "-":
            ret_value = a - b
        if operator == "*":
            ret_value = a * b
        if operator == "**":
            ret_value = a ** b
        if operator == "%":
            ret_value = a % b
        if operator == "nth_root":
            ret_value = a ** (1/b)
        if operator == "min":
            ret_value = min(a, b)
        if operator == "max":
            ret_value = max(a, b)
        if operator == "/":
            if b != 0:
                ret_value = a / b
            else:
                ret_value = 0
        if operator == "//":
            if b != 0:
                ret_value = a // b
            else:
                ret_value = 0

        return (int(ret_value), float(ret_value), str(ret_value))

class StringCondition:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        string_condition_list = ["include", "exclude", "equal"]
        return {"required": {
                "text": ("STRING", {"multiline": False}),
                "condition": (string_condition_list,),
                "sub_string": ("STRING", {"multiline": False}),
            },}

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("output", "string",)
    FUNCTION = 'string_condition'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def string_condition(self, text, condition, sub_string):
        ret = False
        if condition == "include":
            ret = sub_string in text
        if condition == "exclude":
            ret = sub_string not in text
        if condition == "equal":
            ret = text == sub_string
        return (ret, str(ret))


class TextBoxNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "text": ("STRING", {"multiline": True}),
            },}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'text_box_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_box_node(self, text):
        return (text,)

class StringNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "string": ("STRING", {"multiline": False}),
            },}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = 'string_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def string_node(self, string):
        return (string,)

class IntegerNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "int_value":("INT", {"default": 0, "min": -1e18, "max": 1e18, "step": 1}),
            },}

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("int", "string",)
    FUNCTION = 'integer_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def integer_node(self, int_value):
        return (int(int_value), str(int_value))

class FloatNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "float_value":  ("FLOAT", {"default": 0, "min": -1e18, "max": 1e18, "step": 0.00001}),
            },}

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("float", "string",)
    FUNCTION = 'float_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def float_node(self, float_value):
        return (float_value, str(float_value))

class BooleanNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "bool_value": ("BOOLEAN", {"default": False}),
            },}

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("boolean", "string",)
    FUNCTION = 'boolean_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def boolean_node(self, bool_value):
        return (bool_value, str(bool_value))

class IfExecute:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "if_condition": (any,),
                "when_TRUE": (any,),
                "when_FALSE": (any,),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("?",)
    FUNCTION = "if_execute"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def if_execute(self, if_condition, when_TRUE, when_FALSE):
        return (when_TRUE if if_condition else when_FALSE,)

class SwitchCaseNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "switch_condition": ("STRING", {"default": "", "multiline": False}),
                "case_1": ("STRING", {"default": "", "multiline": False}),
                "case_2": ("STRING", {"default": "", "multiline": False}),
                "case_3": ("STRING", {"default": "", "multiline": False}),
                "input_default": (any,),
            },
            "optional": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("?",)
    FUNCTION = "switch_case"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def switch_case(self, switch_condition, case_1, case_2, case_3, input_default, input_1=None, input_2=None, input_3=None):

        output=input_default
        if switch_condition == case_1 and input_1 is not None:
            output=input_1
        elif switch_condition == case_2 and input_2 is not None:
            output=input_2
        elif switch_condition == case_3 and input_3 is not None:
            output=input_3

        return (output,)

class QueueStopNode():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ["stop", "continue"]
        return {
            "required": {
                "any": (any, ),
                "mode": (mode_list,),
                "stop": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = 'stop_node'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def stop_node(self, any, mode,stop):
        if mode == "stop":
            if stop:
                log(f"Queue stopped, it was terminated by node.", "error")
                from comfy.model_management import InterruptProcessingException
                raise InterruptProcessingException()

        return (any,)


class LS_ImageBatchToMultiList:

    def __init__(self):
        self.NODE_NAME = 'ImageBatchToList'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_size": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "image_batch_to_multi_list"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def image_batch_to_multi_list(self, image, batch_size):
        """
        image: [B, H, W, C]
        输出: list of IMAGE batch，每个 batch 大小 <= batch_size
        """
        B = image.shape[0]
        out = []
        sizes = []

        for i in range(0, B, batch_size):
            batch = image[i:i + batch_size]
            out.append(batch)
            sizes.append(batch.shape[0])

        log(f"{self.NODE_NAME}: Convert a batch of {B} images to {sizes}.", message_type='finish')

        return (out,)


class LS_MultiImageListToBatch:

    def __init__(self):
        self.NODE_NAME = 'ImageListToBatch'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    INPUT_IS_LIST = True
    FUNCTION = "multi_image_list_to_batch"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def multi_image_list_to_batch(self, image):
        """
        image: list of IMAGE batch
               每个元素 shape 为 [Bi, Hi, Wi, C]
        输出: 单一 IMAGE batch [sum(Bi), H, W, C]
        """

        # 以第一个 batch 的第一张图作为基准尺寸
        base_h, base_w = image[0].shape[1:3]
        out = []
        sizes =  []

        for batch in image:
            # batch: [B, H, W, C]
            if batch.shape[1:3] != (base_h, base_w):
                # 转成 [B, C, H, W]
                batch = batch.permute(0, 3, 1, 2)

                batch = comfy.utils.common_upscale(
                    batch,
                    base_w,
                    base_h,
                    upscale_method="bicubic",
                    crop="center"
                )

                # 转回 [B, H, W, C]
                batch = batch.permute(0, 2, 3, 1)
            
            sizes.append(batch.shape[0])
            out.append(batch)

        # 沿 batch 维拼接
        out = torch.cat(out, dim=0)
        log(f"{self.NODE_NAME}: Convert {sizes} list(s) to a batch of {out.shape[0]} images.", message_type='finish')

        return (out,)



NODE_CLASS_MAPPINGS = {
    "LayerUtility: QueueStop": QueueStopNode,
    "LayerUtility: SwitchCase": SwitchCaseNode,
    "LayerUtility: If ": IfExecute,
    "LayerUtility: StringCondition": StringCondition,
    "LayerUtility: BooleanOperator": BooleanOperator,
    "LayerUtility: NumberCalculator": NumberCalculator,
    "LayerUtility: BooleanOperatorV2": BooleanOperatorV2,
    "LayerUtility: NumberCalculatorV2": NumberCalculatorV2,
    "LayerUtility: TextBox": TextBoxNode,
    "LayerUtility: String": StringNode,
    "LayerUtility: Integer": IntegerNode,
    "LayerUtility: Float": FloatNode,
    "LayerUtility: Boolean": BooleanNode,
    "LayerUtility: Seed": SeedNode,
    "LayerUtility: ImageBatchToList": LS_ImageBatchToMultiList,
    "LayerUtility: ImageListToBatch": LS_MultiImageListToBatch,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: QueueStop": "LayerUtility: Queue Stop",
    "LayerUtility: SwitchCase": "LayerUtility: Switch Case",
    "LayerUtility: If ": "LayerUtility: If",
    "LayerUtility: StringCondition": "LayerUtility: String Condition",
    "LayerUtility: BooleanOperator": "LayerUtility: Boolean Operator",
    "LayerUtility: NumberCalculator": "LayerUtility: Number Calculator",
    "LayerUtility: BooleanOperatorV2": "LayerUtility: Boolean Operator V2",
    "LayerUtility: NumberCalculatorV2": "LayerUtility: Number Calculator V2",
    "LayerUtility: TextBox": "LayerUtility: TextBox",
    "LayerUtility: String": "LayerUtility: String",
    "LayerUtility: Integer": "LayerUtility: Integer",
    "LayerUtility: Float": "LayerUtility: Float",
    "LayerUtility: Boolean": "LayerUtility: Boolean",
    "LayerUtility: Seed": "LayerUtility: Seed",
    "LayerUtility: ImageBatchToList": "LayerUtility: Image Batch To List(Multi)",
    "LayerUtility: ImageListToBatch": "LayerUtility: Image List To Batch(Multi)",
}