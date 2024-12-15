from .imagefunc import AnyType

anything = AnyType('*')

class LS_AnyRerouter():

    def __init__(self):
        self.NODE_NAME = 'AnyRerouter'


    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "any": (anything, {}),
            },
            "optional": { #
            }
        }

    RETURN_TYPES = (anything,)
    RETURN_NAMES = ('any',)
    FUNCTION = 'any_rerouter'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def any_rerouter(self, any,):
        return (any,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: AnyRerouter": LS_AnyRerouter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: AnyRerouter": "LayerUtility: Any Rerouter"
}