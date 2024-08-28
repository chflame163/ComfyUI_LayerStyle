import contextlib
import copy
import io
import logging
import os
import random

import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
from PIL import Image

"""
This file contains functions to parse RefCOCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_refcoco_json"]


def load_grefcoco_json(
    refer_root,
    dataset_name,
    splitby,
    split,
    image_root,
    extra_annotation_keys=None,
    extra_refer_keys=None,
):
    if dataset_name == "refcocop":
        dataset_name = "refcoco+"
    if dataset_name == "refcoco" or dataset_name == "refcoco+":
        splitby == "unc"
    if dataset_name == "refcocog":
        assert splitby == "umd" or splitby == "google"

    dataset_id = "_".join([dataset_name, splitby, split])

    from .grefer import G_REFER

    logger.info("Loading dataset {} ({}-{}) ...".format(dataset_name, splitby, split))
    logger.info("Refcoco root: {}".format(refer_root))
    timer = Timer()
    refer_root = PathManager.get_local_path(refer_root)
    with contextlib.redirect_stdout(io.StringIO()):
        refer_api = G_REFER(data_root=refer_root, dataset=dataset_name, splitBy=splitby)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(dataset_id, timer.seconds())
        )

    ref_ids = refer_api.getRefIds(split=split)
    img_ids = refer_api.getImgIds(ref_ids)
    refs = refer_api.loadRefs(ref_ids)
    imgs = [refer_api.loadImgs(ref["image_id"])[0] for ref in refs]
    anns = [refer_api.loadAnns(ref["ann_id"]) for ref in refs]
    imgs_refs_anns = list(zip(imgs, refs, anns))

    logger.info(
        "Loaded {} images, {} referring object sets in G_RefCOCO format from {}".format(
            len(img_ids), len(ref_ids), dataset_id
        )
    )

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"] + (extra_annotation_keys or [])
    ref_keys = ["raw", "sent_id"] + (extra_refer_keys or [])

    ann_lib = {}

    NT_count = 0
    MT_count = 0

    for img_dict, ref_dict, anno_dicts in imgs_refs_anns:
        record = {}
        record["source"] = "grefcoco"
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        # Check that information of image, ann and ref match each other
        # This fails only when the data parsing logic or the annotation file is buggy.
        assert ref_dict["image_id"] == image_id
        assert ref_dict["split"] == split
        if not isinstance(ref_dict["ann_id"], list):
            ref_dict["ann_id"] = [ref_dict["ann_id"]]

        # No target samples
        if None in anno_dicts:
            assert anno_dicts == [None]
            assert ref_dict["ann_id"] == [-1]
            record["empty"] = True
            obj = {key: None for key in ann_keys if key in ann_keys}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["empty"] = True
            obj = [obj]

        # Multi target samples
        else:
            record["empty"] = False
            obj = []
            for anno_dict in anno_dicts:
                ann_id = anno_dict["id"]
                if anno_dict["iscrowd"]:
                    continue
                assert anno_dict["image_id"] == image_id
                assert ann_id in ref_dict["ann_id"]

                if ann_id in ann_lib:
                    ann = ann_lib[ann_id]
                else:
                    ann = {key: anno_dict[key] for key in ann_keys if key in anno_dict}
                    ann["bbox_mode"] = BoxMode.XYWH_ABS
                    ann["empty"] = False

                    segm = anno_dict.get("segmentation", None)
                    assert segm  # either list[list[float]] or dict(RLE)
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [
                            poly
                            for poly in segm
                            if len(poly) % 2 == 0 and len(poly) >= 6
                        ]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    ann["segmentation"] = segm
                    ann_lib[ann_id] = ann

                obj.append(ann)

        record["annotations"] = obj

        # Process referring expressions
        sents = ref_dict["sentences"]
        for sent in sents:
            ref_record = record.copy()
            ref = {key: sent[key] for key in ref_keys if key in sent}
            ref["ref_id"] = ref_dict["ref_id"]
            ref_record["sentence"] = ref
            dataset_dicts.append(ref_record)
    #         if ref_record['empty']:
    #             NT_count += 1
    #         else:
    #             MT_count += 1

    # logger.info("NT samples: %d, MT samples: %d", NT_count, MT_count)

    # Debug mode
    # return dataset_dicts[:100]

    return dataset_dicts


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    import sys

    REFCOCO_PATH = "/mnt/lustre/hhding/code/ReLA/datasets"
    COCO_TRAIN_2014_IMAGE_ROOT = "/mnt/lustre/hhding/code/ReLA/datasets/images"
    REFCOCO_DATASET = "grefcoco"
    REFCOCO_SPLITBY = "unc"
    REFCOCO_SPLIT = "train"


    dicts = load_grefcoco_json(
        REFCOCO_PATH,
        REFCOCO_DATASET,
        REFCOCO_SPLITBY,
        REFCOCO_SPLIT,
        COCO_TRAIN_2014_IMAGE_ROOT,
    )
    print(1)
