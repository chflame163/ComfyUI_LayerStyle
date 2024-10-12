from .imagefunc import AnyType
import random


class LSRandomGenerator:

    def __init__(self):
        self.NODE_NAME = 'RandomGenerator'
        self.previous_seeds= set({})
        self.fixed_seed = 0
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0, "min": -1.0e14, "max": 1.0e14, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 10, "min": -1.0e14, "max": 1.0e14, "step": 0.01}),
                "float_decimal_places": ("INT", {"default": 1, "min": 1, "max": 14, "step": 1}),
                "fix_seed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN",)
    RETURN_NAMES = ("int", "float", "bool",)
    FUNCTION = 'random_generator'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def random_generator(self, min_value, max_value, float_decimal_places, fix_seed, image=None):
        batch_size = 1
        if image is not None:
            batch_size = image.shape[0]
        ret_nunbers = []
        for i in range(batch_size):
            new_seed = self.generate_unique_seed()
            if fix_seed:
                if self.fixed_seed == 0:
                    self.fixed_seed = new_seed
                seed = self.fixed_seed
            else:
                seed = new_seed
            random.seed(seed)
            factor = random.uniform(3, 9)
            random_float = random.uniform(min_value, max_value) / factor
            random_float = round(random_float * factor, float_decimal_places)
            random_int = int(random_float)
            random_bool = random_int %2 == 0
            ret_nunbers.append((random_int, random_float, random_bool))

        if len(ret_nunbers) > 1:
            ret_ints = [item[0] for item in ret_nunbers]
            ret_floats = [item[1] for item in ret_nunbers]
            ret_bools = [item[2] for item in ret_nunbers]
            return (ret_ints, ret_floats, ret_bools)
        else:
            return (ret_nunbers[0][0], ret_nunbers[0][1], ret_nunbers[0][2])


    def generate_unique_seed(self) -> int:
        while True:
            new_number = random.randint(0, 1e14)
            if new_number not in self.previous_seeds:
                self.previous_seeds.add(new_number)
                return new_number

class LS_RandomGeneratorV2:

    def __init__(self):
        self.NODE_NAME = 'RandomGeneratorV2'
        self.previous_seeds= set({})
        self.fixed_seed = 0
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0, "min": -1.0e14, "max": 1.0e14, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 10, "min": -1.0e14, "max": 1.0e14, "step": 0.01}),
                "least": ("FLOAT", {"default": 0, "min": 0, "max": 1.0e14, "step": 0.01}),
                "float_decimal_places": ("INT", {"default": 1, "min": 1, "max": 14, "step": 1}),
                "seed":("INT", {"default": 0, "min": 0, "max": 1e14, "step": 1}),

            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN",)
    RETURN_NAMES = ("int", "float", "bool",)
    # OUTPUT_IS_LIST = (True, True, True,)
    FUNCTION = 'random_generator_v2'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def random_generator_v2(self, min_value, max_value, least, float_decimal_places, seed, image=None):
        batch_size = 1
        if image is not None:
            batch_size = image.shape[0]
        ret_nunbers = []
        for i in range(batch_size):

            random.seed(seed)
            max_loop = 500
            i = 0
            while i < max_loop:
                new_number = random.uniform(min_value, max_value)
                if abs(new_number) - least >= 0 or least > max_value:
                    break
                i += 1

            # 转浮点
            factor = random.uniform(3, 9)
            random_float = new_number / factor
            random_float = round(random_float * factor, float_decimal_places)
            random_int = int(random_float)
            random_bool = random_int %2 == 0
            ret_nunbers.append((random_int, random_float, random_bool))

        if len(ret_nunbers) > 1:
            ret_ints = [item[0] for item in ret_nunbers]
            ret_floats = [item[1] for item in ret_nunbers]
            ret_bools = [item[2] for item in ret_nunbers]
            return (ret_ints, ret_floats, ret_bools)
        else:
            return (ret_nunbers[0][0], ret_nunbers[0][1], ret_nunbers[0][2])


NODE_CLASS_MAPPINGS = {
    "LayerUtility: RandomGenerator": LSRandomGenerator,
    "LayerUtility: RandomGeneratorV2": LS_RandomGeneratorV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: RandomGenerator": "LayerUtility: Random Generator",
    "LayerUtility: RandomGeneratorV2": "LayerUtility: Random Generator V2"
}