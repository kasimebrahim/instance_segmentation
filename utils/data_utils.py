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

#########################################################
# Edit annotations of json in main_path with reference json
# file in ref_path.
# This is usefull to convetert an annotated dataset with a
# wrong image format. When the image format and size change
# we need to applay the neccessary edit in the annotations.
# And this will help with that.
#
# USAGE:
# step1:- convert the images to jpg format with an external
#         tool.
# step2:- Import the converted images to the VIA-annotator
#         and export json annotations with empty annotation.
#         The json will contain the right file-name and size.
# step3:- Call this function with the original annotation and
#         the above new empty annotation json as a reference.
# The output json will contain the right annotation for
# the converted images.
#########################################################
def label_rename(main_path, ref_path):
    main = json.load(open(main_path))
    ref = json.load(open(ref_path))

    main_k = list(main.keys())
    ref_k = list(ref.keys())

    for m_k in main_k:
        m_name, ext = m_k.rsplit(".", 1)
        if "png" in ext:
            r_k = findkey(m_name, ref_k)
            main[m_k]["filename"] = ref[r_k]["filename"]
            main[m_k]["size"] = ref[r_k]["size"]
            main[r_k] = main.pop(m_k)

    with open(main_path + "_edited", 'w') as out:
        json.dump(main, out, sort_keys=True, indent=4)

def findkey(pat, keys):
    for m_k in keys:
        m_name, ext = m_k.rsplit(".", 1)
        if pat == m_name:
            return m_k
    raise ValueError("Key not found in reference json!");

#########################################################
# Split Data in dir into two.
#########################################################
def split_data(dir, size):
    main = json.load(open(os.path.join(dir, "labels.json")))
    keys = list(main.keys())

    part = {}
    t_path = os.path.join(dir, "train")
    v_path = os.path.join(dir, "validation")

    try:
        os.mkdir(t_path)
        os.mkdir(v_path)
    except OSError:
        print ("Creation of the directory failed")

    for n in range(size):
        k = random.choice(keys)
        part[k] = main.pop(k)
        shutil.move(os.path.join(dir, part[k]["filename"]),
                        os.path.join(v_path, part[k]["filename"]))
        keys.remove(k)

    for k in keys:
        shutil.move(os.path.join(dir, main[k]["filename"]),
                        os.path.join(t_path, main[k]["filename"]))

    with open(os.path.join(t_path, "labels.json"), 'w') as out:
        json.dump(main, out, sort_keys=True, indent=4)

    with open(os.path.join(v_path, "labels.json"), 'w') as out:
        json.dump(part, out, sort_keys=True, indent=4)
