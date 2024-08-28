"""
grefer v0.1
This interface provides access to gRefCOCO.

The following API functions are defined:
G_REFER      - REFER api class
getRefIds    - get ref ids that satisfy given filter conditions.
getAnnIds    - get ann ids that satisfy given filter conditions.
getImgIds    - get image ids that satisfy given filter conditions.
getCatIds    - get category ids that satisfy given filter conditions.
loadRefs     - load refs with the specified ref ids.
loadAnns     - load anns with the specified ann ids.
loadImgs     - load images with the specified image ids.
loadCats     - load category names with the specified category ids.
getRefBox    - get ref's bounding box [x, y, w, h] given the ref_id
showRef      - show image, segmentation or box of the referred object with the ref
getMaskByRef - get mask and area of the referred object given ref or ref ids
getMask      - get mask and area of the referred object given ref
showMask     - show mask of the referred object given ref
"""

import itertools
import json
import os.path as osp
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pycocotools import mask


class G_REFER:
    def __init__(self, data_root, dataset="grefcoco", splitBy="unc"):
        # provide data_root folder which contains grefcoco
        print("loading dataset %s into memory..." % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ["grefcoco"]:
            self.IMAGE_DIR = osp.join(data_root, "images/train2014")
        else:
            raise KeyError("No refer dataset is called [%s]" % dataset)

        tic = time.time()

        # load refs from data/dataset/refs(dataset).json
        self.data = {}
        self.data["dataset"] = dataset

        ref_file = osp.join(self.DATA_DIR, f"grefs({splitBy}).p")
        if osp.exists(ref_file):
            self.data["refs"] = pickle.load(open(ref_file, "rb"), fix_imports=True)
        else:
            ref_file = osp.join(self.DATA_DIR, f"grefs({splitBy}).json")
            if osp.exists(ref_file):
                self.data["refs"] = json.load(open(ref_file, "rb"))
            else:
                raise FileNotFoundError("JSON file not found")

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, "instances.json")
        instances = json.load(open(instances_file, "r"))
        self.data["images"] = instances["images"]
        self.data["annotations"] = instances["annotations"]
        self.data["categories"] = instances["categories"]

        # create index
        self.createIndex()
        print("DONE (t=%.2fs)" % (time.time() - tic))

    @staticmethod
    def _toList(x):
        return x if isinstance(x, list) else [x]

    @staticmethod
    def match_any(a, b):
        a = a if isinstance(a, list) else [a]
        b = b if isinstance(b, list) else [b]
        return set(a) & set(b)

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print("creating index...")
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        Anns[-1] = None
        for ann in self.data["annotations"]:
            Anns[ann["id"]] = ann
            imgToAnns[ann["image_id"]] = imgToAnns.get(ann["image_id"], []) + [ann]
        for img in self.data["images"]:
            Imgs[img["id"]] = img
        for cat in self.data["categories"]:
            Cats[cat["id"]] = cat["name"]

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        availableSplits = []
        for ref in self.data["refs"]:
            # ids
            ref_id = ref["ref_id"]
            ann_id = ref["ann_id"]
            category_id = ref["category_id"]
            image_id = ref["image_id"]

            if ref["split"] not in availableSplits:
                availableSplits.append(ref["split"])

            # add mapping related to ref
            if ref_id in Refs:
                print("Duplicate ref id")
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]

            category_id = self._toList(category_id)
            added_cats = []
            for cat in category_id:
                if cat not in added_cats:
                    added_cats.append(cat)
                    catToRefs[cat] = catToRefs.get(cat, []) + [ref]

            ann_id = self._toList(ann_id)
            refToAnn[ref_id] = [Anns[ann] for ann in ann_id]
            for ann_id_n in ann_id:
                annToRef[ann_id_n] = annToRef.get(ann_id_n, []) + [ref]

            # add mapping of sent
            for sent in ref["sentences"]:
                Sents[sent["sent_id"]] = sent
                sentToRef[sent["sent_id"]] = ref
                sentToTokens[sent["sent_id"]] = sent["tokens"]

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        self.availableSplits = availableSplits
        print("index created.")

    def getRefIds(self, image_ids=[], cat_ids=[], split=[]):
        image_ids = self._toList(image_ids)
        cat_ids = self._toList(cat_ids)
        split = self._toList(split)

        for s in split:
            if s not in self.availableSplits:
                raise ValueError(f"Invalid split name: {s}")

        refs = self.data["refs"]

        if len(image_ids) > 0:
            lists = [self.imgToRefs[image_id] for image_id in image_ids]
            refs = list(itertools.chain.from_iterable(lists))
        if len(cat_ids) > 0:
            refs = [ref for ref in refs if self.match_any(ref["category_id"], cat_ids)]
        if len(split) > 0:
            refs = [ref for ref in refs if ref["split"] in split]

        ref_ids = [ref["ref_id"] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], ref_ids=[]):
        image_ids = self._toList(image_ids)
        ref_ids = self._toList(ref_ids)

        if any([len(image_ids), len(ref_ids)]):
            if len(image_ids) > 0:
                lists = [
                    self.imgToAnns[image_id]
                    for image_id in image_ids
                    if image_id in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data["annotations"]
            ann_ids = [ann["id"] for ann in anns]
            if len(ref_ids) > 0:
                lists = [self.Refs[ref_id]["ann_id"] for ref_id in ref_ids]
                anns_by_ref_id = list(itertools.chain.from_iterable(lists))
                ann_ids = list(set(ann_ids).intersection(set(anns_by_ref_id)))
        else:
            ann_ids = [ann["id"] for ann in self.data["annotations"]]

        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = self._toList(ref_ids)

        if len(ref_ids) > 0:
            image_ids = list(set([self.Refs[ref_id]["image_id"] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        return [self.Refs[ref_id] for ref_id in self._toList(ref_ids)]

    def loadAnns(self, ann_ids=[]):
        if isinstance(ann_ids, str):
            ann_ids = int(ann_ids)
        return [self.Anns[ann_id] for ann_id in self._toList(ann_ids)]

    def loadImgs(self, image_ids=[]):
        return [self.Imgs[image_id] for image_id in self._toList(image_ids)]

    def loadCats(self, cat_ids=[]):
        return [self.Cats[cat_id] for cat_id in self._toList(cat_ids)]

    def getRefBox(self, ref_id):
        anns = self.refToAnn[ref_id]
        return [ann["bbox"] for ann in anns]  # [x, y, w, h]

    def showRef(self, ref, seg_box="seg"):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref["image_id"]]
        I = io.imread(osp.join(self.IMAGE_DIR, image["file_name"]))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref["sentences"]):
            print("%s. %s" % (sid + 1, sent["sent"]))
        # show segmentations
        if seg_box == "seg":
            ann_id = ref["ann_id"]
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = "none"
            if type(ann["segmentation"][0]) == list:
                # polygon used for refcoco*
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=(1, 1, 0, 0),
                    linewidths=3,
                    alpha=1,
                )
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=(1, 0, 0, 0),
                    linewidths=1,
                    alpha=1,
                )
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann["segmentation"]
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == "box":
            ann_id = ref["ann_id"]
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref["ref_id"])
            box_plot = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor="green",
                linewidth=3,
            )
            ax.add_patch(box_plot)

    def getMask(self, ann):
        if not ann:
            return None
        if ann["iscrowd"]:
            raise ValueError("Crowd object")
        image = self.Imgs[ann["image_id"]]
        if type(ann["segmentation"][0]) == list:  # polygon
            rle = mask.frPyObjects(ann["segmentation"], image["height"], image["width"])
        else:
            rle = ann["segmentation"]

        m = mask.decode(rle)
        m = np.sum(
            m, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {"mask": m, "area": area}

    def getMaskByRef(self, ref=None, ref_id=None, merge=False):
        if not ref and not ref_id:
            raise ValueError
        if ref:
            ann_ids = ref["ann_id"]
            ref_id = ref["ref_id"]
        else:
            ann_ids = self.getAnnIds(ref_ids=ref_id)

        if ann_ids == [-1]:
            img = self.Imgs[self.Refs[ref_id]["image_id"]]
            return {
                "mask": np.zeros([img["height"], img["width"]], dtype=np.uint8),
                "empty": True,
            }

        anns = self.loadAnns(ann_ids)
        mask_list = [self.getMask(ann) for ann in anns if not ann["iscrowd"]]

        if merge:
            merged_masks = sum([mask["mask"] for mask in mask_list])
            merged_masks[np.where(merged_masks > 1)] = 1
            return {"mask": merged_masks, "empty": False}
        else:
            return mask_list

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M["mask"]
        ax = plt.gca()
        ax.imshow(msk)
