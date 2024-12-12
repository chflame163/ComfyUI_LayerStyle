import importlib.util
import os
import sys
import json

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

python = sys.executable

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def serialize(obj):
    if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
        return obj
    return str(obj)  # 转为字符串


py = get_ext_dir("py")
files = os.listdir(py)
all_nodes = {}
for file in files:
    if not file.endswith(".py"):
        continue
    name = os.path.splitext(file)[0]
    imported_module = importlib.import_module(".py.{}".format(name), __name__)
    try:
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}
        serialized_CLASS_MAPPINGS = {k: serialize(v) for k, v in imported_module.NODE_CLASS_MAPPINGS.items()}
        serialized_DISPLAY_NAME_MAPPINGS = {k: serialize(v) for k, v in imported_module.NODE_DISPLAY_NAME_MAPPINGS.items()}
        all_nodes[file]={"NODE_CLASS_MAPPINGS": serialized_CLASS_MAPPINGS, "NODE_DISPLAY_NAME_MAPPINGS": serialized_DISPLAY_NAME_MAPPINGS}
    except:
        pass


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
