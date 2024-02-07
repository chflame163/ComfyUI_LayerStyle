import os
import glob
from .imagefunc import *

# lut_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'lut')
# ini_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "resource_dir.ini")
#
# try:
#     with open(ini_file, 'r') as f:
#         ini = f.readlines()
#         for line in ini:
#             if line.startswith('LUT_dir='):
#                 d = line[line.find('=') + 1:].rstrip().lstrip()
#                 break
#         if os.path.exists(d):
#             lut_dir = d
#         else:
#             log(f'ERROR: invalid LUT dir, default to be used. check {ini_file}')
# except Exception as e:
#     log(f'ERROR: {ini_file} ' + repr(e))
#
# file_list = glob.glob(lut_dir + '/*.cube')
# lut_dict = {}
# for i in range(len(file_list)):
#     _, filename =  os.path.split(file_list[i])
#     lut_dict[filename] = file_list[i]
# lut_list = list(lut_dict.keys())
# log(f'find {len(lut_list)} LUTs in {lut_dir}')

class ColorCorrectLUTapply:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "LUT": (LUT_LIST,),  # LUT文件
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LUTapply'
    CATEGORY = '😺dzNodes/LayerColor'
    OUTPUT_NODE = True

    def color_correct_LUTapply(self, image, LUT):

        _image = tensor2pil(image)
        lut_file = LUT_DICT[LUT]
        ret_image = lut_apply(_image, lut_file)

        return (pil2tensor(ret_image),)

NODE_CLASS_MAPPINGS = {
    "LayerColor: LUT Apply": ColorCorrectLUTapply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerColor: LUT Apply": "LayerColor: LUT Apply"
}