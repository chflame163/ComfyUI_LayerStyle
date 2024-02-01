import json
import torch
from .imagefunc import tensor2pil, log
from .imagefunc import AnyType

any = AnyType("*")

class PrintInfo:

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "anything": (any, {}),
      },
    }

  CATEGORY = '😺dzNodes/LayerUtility'
  RETURN_TYPES = ()
  FUNCTION = "print_info"
  OUTPUT_NODE = True

  def print_info(self, anything=None):
    value = 'PrintInfo:\nInput type is: ' + str(type(anything)) + '\n'
    if isinstance(anything, torch.Tensor):
      image = tensor2pil(anything)
      value = value + 'Image.size=' + str(image.size) + ', Image.mode=' + str(image.mode) + '\n'
    if anything is not None:
      try:
        value = value + json.dumps(anything) + "\n"
      except Exception:
        try:
          value = value + str(anything) + "\n"
        except Exception:
          value = 'source exists, but could not be serialized.'
    log(value)
    return {"ui": {"text": (value,)}}

NODE_CLASS_MAPPINGS = {
    "LayerUtility: PrintInfo": PrintInfo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PrintInfo": "LayerUtility: PrintInfo"
}