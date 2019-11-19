"""
loader.py
This extends mask-rcnn Dataset. Handles DataSet loading.

Author kasim <se.kasim.ebrahim@gmail.com>
"""

import os
import json
import skimage
import hashlib
import numpy as np
import sys

LIBS = 'libs/'
if LIBS not in sys.path:
    sys.path.append(LIBS)

ROOT_DIR = os.path.abspath(".")

from mrcnn import utils


class SegmentationDataSet(utils.Dataset):

    def load_doc_seg(self, dir, type):
        #Data has to be validation or training.
        assert type in["train", "validation"]

        dataset_dir = os.path.join(dir, type)

        segments = ["title", "subtitle", "paragraph", "footnotes",
                    "header", "footer", "page", "signature"]
        for i in range(len(segments)):
            self.add_class("doc_seg", i, segments[i])

        annotations = list(json.load(open(os.path.join(dataset_dir, "labels.json"))).
                          values())

        for annotation in annotations:
            bounding_boxs = [box['shape_attributes'] for box in annotation['regions'].values()]
            classes = [region['region_attributes'] for region in annotation['regions'].values()]
            image = skimage.io.imread(os.path.join(dataset_dir, annotation['filename']))
            height, width = image.shape[:2]

            self.add_image(
                "doc_seg",
                image_id=annotation['filename'],
                path=os.path.join(dataset_dir, annotation['filename']),
                height=height,
                width=width,
                bounding_boxs=bounding_boxs,
                classes=classes,
                class_label="type"
            )

    def hash(self, string):
        return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16) % (10 ** 8)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        assert image_info['source'] == 'doc_seg'

        image_mask = np.zeros(
            [image_info['height'], image_info['width'], len(image_info['bounding_boxs'])],
            dtype=np.uint8
        )

        for i, box in enumerate(image_info['bounding_boxs']):
            start = (box['y'], box['x'])
            end = (start[0]+box['height']-1, start[1]+box['width']-1)
            rr, cc = skimage.draw.rectangle(start=start, end=end,
                     shape=(image_info["height"], image_info["width"]))

            image_mask[rr, cc, i] = 1

        class_ids = np.array([self.class_names.index(c["type"]) for c in image_info['classes']])
        return image_mask.astype(np.bool), class_ids.astype(np.int32)

class PubLayNetDataSet(utils.Dataset):

    def load_doc_seg(self, dir, type):
        #Data has to be validation or training.
        assert type in["train", "validation"]

        dataset_dir = os.path.join(dir, type)

        segments = ["text", "title", "list", "table",
                    "figure"]
        for i in range(len(segments)):
            self.add_class("pul_lay_seg", i, segments[i])

        annotations = list(json.load(open(os.path.join(dataset_dir, "labels.json"))).
                          values())

        for annotation in annotations:
            bounding_boxs = [box['shape_attributes'] for box in annotation['regions'].values()]
            classes = [region['region_attributes'] for region in annotation['regions'].values()]
            image = skimage.io.imread(os.path.join(dataset_dir, annotation['filename']))
            height, width = image.shape[:2]

            self.add_image(
                "pul_lay_seg",
                image_id=annotation['filename'],
                path=os.path.join(dataset_dir, annotation['filename']),
                height=height,
                width=width,
                bounding_boxs=bounding_boxs,
                classes=classes,
                class_label="type"
            )

    def hash(self, string):
        return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16) % (10 ** 8)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        assert image_info['source'] == 'pul_lay_seg'

        image_mask = np.zeros(
            [image_info['height'], image_info['width'], len(image_info['bounding_boxs'])],
            dtype=np.uint8
        )

        for i, box in enumerate(image_info['bounding_boxs']):
            start = (box['y'], box['x'])
            end = (start[0]+box['height']-1, start[1]+box['width']-1)
            rr, cc = skimage.draw.rectangle(start=start, end=end,
                     shape=(image_info["height"], image_info["width"]))

            image_mask[rr, cc, i] = 1

        class_ids = np.array([self.class_names.index(c["type"]) for c in image_info['classes']])
        return image_mask.astype(np.bool), class_ids.astype(np.int32)
