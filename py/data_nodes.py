from .imagefunc import AnyType

any = AnyType("*")


class BooleanOperator:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["==", "!=", "and", "or", "xor", "not(a)"]
        return {"required": {
                "a": (any, {}),
                "b": (any, {}),
                "operator": (operator_list,),
            },}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = 'bool_operator_node'
    CATEGORY = '😺dzNodes/LayerUtility'

    def bool_operator_node(self, a, b, operator):
        ret_value = False
        if operator == "==":
            ret_value = a == b
        if operator == "!=":
            ret_value = a != b
        if operator == "and":
            ret_value = a and b
        if operator == "or":
            ret_value = a or b
        if operator == "xor":
            ret_value = not(a == b)
        if operator == "not(a)":
            ret_value = not a

        return (ret_value,)

class NumberCalculator:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["+", "-", "*", "/", "**", "//", "%" ]
        return {"required": {
                "a": (any, {}),
                "b": (any, {}),
                "operator": (operator_list,),
            },}

    RETURN_TYPES = ("INT", "FLOAT",)
    RETURN_NAMES = ("int", "float",)
    FUNCTION = 'number_calculator_node'
    CATEGORY = '😺dzNodes/LayerUtility'

    def number_calculator_node(self, a, b, operator):
        ret_value = 0
        if operator == "+":
            ret_value = a + b
        if operator == "-":
            ret_value = a - b
        if operator == "*":
            ret_value = a * b
        if operator == "/":
            ret_value = a / b
        if operator == "**":
            ret_value = a ** b
        if operator == "//":
            ret_value = a // b
        if operator == "%":
            ret_value = a % b

        return (int(ret_value), float(ret_value),)

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
    CATEGORY = '😺dzNodes/LayerUtility'

    def text_box_node(self, text):
        return (text,)

class IntegerNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "int_value":("INT", {"default": 0, "min": -99999999999999999999, "max": 99999999999999999999, "step": 1}),
            },}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = 'integer_node'
    CATEGORY = '😺dzNodes/LayerUtility'

    def integer_node(self, int_value):
        return (int_value,)

class FloatNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "float_value":  ("FLOAT", {"default": 0, "min": -99999999999999999999, "max": 99999999999999999999, "step": 0.00001}),
            },}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = 'float_node'
    CATEGORY = '😺dzNodes/LayerUtility'

    def float_node(self, float_value):
        return (float_value,)

class BooleanNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "bool_value": ("BOOLEAN", {"default": False}),
            },}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = 'boolean_node'
    CATEGORY = '😺dzNodes/LayerUtility'

    def boolean_node(self, bool_value):
        return (bool_value,)




NODE_CLASS_MAPPINGS = {
    "LayerUtility: BooleanOperator": BooleanOperator,
    "LayerUtility: NumberCalculator": NumberCalculator,
    "LayerUtility: TextBox": TextBoxNode,
    "LayerUtility: Integer": IntegerNode,
    "LayerUtility: Float": FloatNode,
    "LayerUtility: Boolean": BooleanNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: BooleanOperator": "LayerUtility: BooleanOperator",
    "LayerUtility: NumberCalculator": "LayerUtility: NumberCalculator",
    "LayerUtility: TextBox": "LayerUtility: TextBox",
    "LayerUtility: Integer": "LayerUtility: Integer",
    "LayerUtility: Float": "LayerUtility: Float",
    "LayerUtility: Boolean": "LayerUtility: Boolean"
}