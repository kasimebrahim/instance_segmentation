"""
Segmentation.py

Author kasim <se.kasim.ebrahim@gmail.com>
"""

import os
import skimage
import json
import datetime
import imgaug
import numpy as np
import Loader as ld
import sys

LIBS = 'libs/'
if LIBS not in sys.path:
    sys.path.append(LIBS)

from mrcnn.config import Config
from mrcnn import model as modellib

ROOT_DIR = os.path.abspath(".")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class SegmentationConfig(Config):

    NAME="doc_seg"
    IMAGES_PER_GPU=2
    NUM_CLASSES=1 + 8
    STEPS_PER_EPOCH=100
    DETECTION_MIN_CONFIDENCE = 0.8
    BACKBONE="resnet50"
    # BACKBONE="resnet101"
    # Weight decay regularization
    # WEIGHT_DECAY = 0.01

class InferenceConfig(SegmentationConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU=1
    BACKBONE="resnet50"
    # BACKBONE="resnet101"

def train(model):
    datasets_training = ld.SegmentationDataSet()
    datasets_validation = ld.SegmentationDataSet()

    datasets_training.load_doc_seg(args.datasets, "train")
    datasets_validation.load_doc_seg(args.datasets, "validation")

    datasets_training.prepare()
    datasets_validation.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)
    model.train(datasets_training, datasets_validation, config.LEARNING_RATE,
                epochs=100, layers='all', augmentation=augmentation)

    # model.train(datasets_training, datasets_validation, config.LEARNING_RATE,
    #             epochs=100, layers='heads')

def infer(model, data_dir):
    SEGMENTS = ["BG", "title", "subtitle", "paragraph", "footnotes",
                "header", "footer", "page", "signature"]

    from collections import defaultdict
    documents = defaultdict(list)
    for doc, page, path in gen_image_path(data_dir):
        print("\nDetecting : {}\n".format(path))
        image = skimage.io.imread(path)
        results = model.detect([image], verbose=1)[0]
        page = {'page':page, 'elements':[]}

        ###################
        for c, b in zip(results['class_ids'], results['rois']):
            page['elements'].append({
                    'type': SEGMENTS[c],
                    'y1':str(b[0]),
                    'x1':str(b[1]),
                    'y2':str(b[2]),
                    'x2':str(b[3])})

        documents[doc].append(page)
    # write to file
    with open('documents.json', 'w') as out:
        json.dump(documents, out, sort_keys=True)

########################################################
# generate document_name, page, path of images from
# data_set directory. supports two types of date set
# arrengements.
#
# One: Each documenat is in its own directory and every
#      page of the document is in the directory.
#      i:e ds/doc/p01.jpg -> (doc, p01, datasets/doc/p01.jpg)
# Two: All pages of all the documents are under one
#      directory. And every page is named as document_name
#      concatinated with page name/number.
#      i:e ds/doc_p01.jpg -> (doc, p01, ds/doc_p01.jpg)
########################################################
def gen_image_path(parent_dir):
    path, dirs, files = next(os.walk(parent_dir))
    # if wrapped by a directory then the dir_name is the
    # document name and the file name is the page name.
    for dir in dirs:
        new_path, _,_files = next(os.walk(os.path.join(path, dir)))
        for file in _files:
            # document, page, path_to_image
            yield (dir, file.rsplit(".", 1)[0], os.path.join(new_path, file))
    # if not wrapped by a directory then split the file_name
    # to document name and file name.
    for file in files:
        doc, page = file.rsplit(".", 1)[0].rsplit("_",1)
        yield (doc, page, os.path.join(path, file))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train Contract Documents Segmentation")
    parser.add_argument("command", metavar="<command>", help="'train' or 'segment'")
    parser.add_argument("--datasets", required=False, metavar="path\\to\\DocSeg\\datasets",
                        help="Path to your datasets")
    parser.add_argument("--model", required=False, metavar="path\\to\\model.h5",
                        help="Path to trained model")
    parser.add_argument("--log", required=False, default=DEFAULT_LOGS_DIR,
                        metavar="path\\to\\log\\folder",
                        help="Path to folder to save log and models")
    parser.add_argument("--image", required=False, metavar="path\\to\\image",
                        help="Path to image to segment")
    parser.add_argument("--pickup", required=False, help="pickup from last")
    args = parser.parse_args()

    if args.command == "train":
        assert args.datasets, "Path to datasets to train is required."
        config = SegmentationConfig()

    elif args.command == "segment":
        assert args.image, "Path to image to segment is required."
        config = InferenceConfig()

    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.log)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.log)

    if args.pickup == "true":
        weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)

    if args.command=="train":
        if args.model:
            print("load-weights")
            model_path = args.model
            model.load_weights(model_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        train(model)
    else:
        model_path = args.model
        model.load_weights(model_path, by_name=True)
        infer(model, args.image)
