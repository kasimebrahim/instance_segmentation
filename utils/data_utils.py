"""
data_utils.py

Author kasim <se.kasim.ebrahim@gmail.com>
"""

import os
import json
import errno
import shutil
import random

def gen_file_path(parent_dir, ext=""):
    path, dirs, files = next(os.walk(parent_dir))
    # if wrapped by a directory then the dir_name is the
    # document name and the file name is the page name.
    for dir in dirs:
        new_path, _,_files = next(os.walk(os.path.join(path, dir)))
        for file in _files:
            # document, page, path_to_image
            page, _ext = file.rsplit(".", 1)
            if ext and (_ext != ext):
                continue
            yield (dir, page, os.path.join(new_path, file))
    # if not wrapped by a directory then split the file_name
    # to document name and file name.
    for file in files:
        name, _ext = file.rsplit(".", 1)
        if ext and (_ext != ext):
            continue
        try:
            doc, page = name.rsplit("-",1)
        except:
            yield ("", "", os.path.join(path, file))
            continue
        yield (doc, page, os.path.join(path, file))

#########################################################
# Merge all json files containing VIA annotations in dir.
# And store in sub_dir.
#########################################################
def merge_json(dir, sub_dir):
    data={}
    for tp in gen_file_path(dir, ext="json"):
        data.update(json.load(open(tp[2])))

    if not os.path.exists(sub_dir):
        try:
            os.makedirs(sub_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(os.path.join(sub_dir, "labels.json"), 'w') as out:
        json.dump(data, out, sort_keys=True, indent=4)

    return sub_dir
    
##############################################################
# Merge all images of jpg extenssion in dir,and store in sub_dir.
##############################################################
def merge_images(dir, sub_dir):
    if not os.path.exists(sub_dir):
        try:
            os.makedirs(sub_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    for tp in gen_file_path(dir, ext="jpg"):
        shutil.copyfile(tp[2], os.path.join(sub_dir, tp[1]+'.jpg'))

#########################################################
# Merge all separate datasets in dir in to sub_dir.
#########################################################
def prep_data(dir, sub_dir):
    merge_json(dir, sub_dir)
    merge_images(dir, sub_dir)
