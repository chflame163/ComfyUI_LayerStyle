


class CropBoxResolve:

    def __init__(self):
        self.NODE_NAME = 'CropBoxResolve'

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "crop_box": ("BOX",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x", "y", "width", "height")
    FUNCTION = 'crop_box_resolve'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_box_resolve(self, crop_box
                  ):

        (x1, y1, x2, y2) = crop_box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        return (x1, y1, x2 - x1, y2 - y1,)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: CropBoxResolve": CropBoxResolve
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropBoxResolve": "LayerUtility: CropBoxResolve"
}