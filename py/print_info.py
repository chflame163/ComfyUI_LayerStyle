import torch

from .imagefunc import *

any = AnyType("*")

class PrintInfo:

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "anything": (any, {}),
      },
    }

  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("text",)
  FUNCTION = "print_info"
  CATEGORY = '😺dzNodes/LayerUtility/Data'
  OUTPUT_NODE = True

  def print_info(self, anything=None):
    value = f'PrintInfo:\nInput type = {type(anything)}'
    if isinstance(anything, torch.Tensor):
      value += f"\n Input dim = {anything.dim()}, shape[0] = {anything.shape[0]} \n"
      for i in range(anything.shape[0]):
        t = anything[i]
        image = tensor2pil(t)
        value += f'\n index {i}: Image.size = {image.size}, Image.mode = {image.mode}, dim = {t.dim()}, '
        for j in range(t.dim()):
          value += f'shape[{j}] = {t.shape[j]}, '
        # value += f'\n {t} \n'
    elif anything is not None:
      try:
        value = value + json.dumps(anything) + "\n"
      except Exception:
        try:
          value = value + str(anything) + "\n"
        except Exception:
          value = 'source exists, but could not be serialized.'
    else:
      value = 'source does not exist.'

    log(value)

    return (value,)

NODE_CLASS_MAPPINGS = {
    "LayerUtility: PrintInfo": PrintInfo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PrintInfo": "LayerUtility: PrintInfo"
}