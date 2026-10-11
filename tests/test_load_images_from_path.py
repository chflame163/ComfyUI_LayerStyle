import importlib.util
import mimetypes
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image


folder_paths = types.ModuleType("folder_paths")
folder_paths.input_directory = None


def filter_files_content_types(files, content_types):
    return [
        name
        for name in files
        if (mimetypes.guess_type(name)[0] or "").split("/", 1)[0] in content_types
    ]


def get_annotated_filepath(name, default_dir=None):
    root = os.path.realpath(default_dir or folder_paths.input_directory)
    target = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath((root, target)) != root:
        raise ValueError(f"Invalid file path: {name!r}")
    return target


folder_paths.filter_files_content_types = filter_files_content_types
folder_paths.get_annotated_filepath = get_annotated_filepath
sys.modules["folder_paths"] = folder_paths

node_helpers = types.ModuleType("node_helpers")
node_helpers.pillow = lambda function, *args: function(*args)
sys.modules["node_helpers"] = node_helpers

module_path = Path(__file__).parents[1] / "py" / "load_images_from_path.py"
spec = importlib.util.spec_from_file_location("load_images_from_path", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LoadImagesFromPathTest(unittest.TestCase):
    def test_external_folder_uses_selected_directory(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as input_directory:
            folder_paths.input_directory = input_directory
            Image.new("RGB", (1, 1)).save(Path(directory) / "image.png")

            result = module.LS_LoadImagesFromPath().ls_load_images(
                directory, image_load_cap=0, select_every_nth=1
            )

            self.assertEqual(result[2], ["image.png"])
            self.assertEqual(result[3], 1)

    def test_filters_labels_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            folder_paths.input_directory = directory
            path = Path(directory)
            Image.new("RGB", (1, 1)).save(path / "one.png")
            (path / "one.txt").write_text("label")
            Image.new("RGB", (1, 1)).save(path / "two.png")

            with mock.patch.object(
                module.os, "listdir", return_value=["one.png", "one.txt", "two.png"]
            ):
                result = module.LS_LoadImagesFromPath().ls_load_images(
                    directory, image_load_cap=0, select_every_nth=2
                )

            self.assertEqual(result[2], ["one.png"])
            self.assertEqual(result[3], 1)

    def test_selected_folder_containment_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            folder_paths.input_directory = directory
            outside_image = Path(outside) / "outside.png"
            Image.new("RGB", (1, 1)).save(outside_image)
            os.symlink(outside_image, Path(directory) / "linked.png")

            with self.assertRaisesRegex(ValueError, "Invalid file path"):
                module.LS_LoadImagesFromPath().ls_load_images(
                    directory, image_load_cap=0, select_every_nth=1
                )


if __name__ == "__main__":
    unittest.main()
