#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import json
import numpy as np
from collections import OrderedDict
from utils.files_utils import subdirs, subfiles


def classes_dice_mean(filename):
    with open(filename, 'r') as f:
        res = json.load(f)
    class_ids = np.array([int(i) for i in res['results']['mean'].keys() if (i != 'mean')])
    class_ids = class_ids[class_ids != 0]
    class_ids = class_ids[class_ids != -1]
    class_ids = class_ids[class_ids != 99]

    tmp = res['results']['mean'].get('99')
    if tmp is not None:
        _ = res['results']['mean'].pop('99')

    metrics = res['results']['mean']['1'].keys()
    res['results']['mean']["mean"] = OrderedDict()
    for m in metrics:
        foreground_values = [res['results']['mean'][str(i)][m] for i in class_ids]
        res['results']['mean']["mean"][m] = np.nanmean(foreground_values)
    with open(filename, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)

def run_in_folder(folder):
    json_files = subfiles(folder, True, None, ".json", True)
    json_files = [i for i in json_files if not i.split("/")[-1].startswith(".") and not i.endswith("_globalMean.json")] # stupid mac
    for j in json_files:
        classes_dice_mean(j)
        
def recur_copy(current_folder, out_folder, prefix="simon_", suffix="ummary.json"):
    """
    Recursively run through all subfolders of current_folder and copy all files that end with
    suffix with some automatically generated prefix into out_folder
    :param current_folder:
    :param out_folder:
    :param prefix:
    :return:
    """
    s = subdirs(current_folder, join=False)
    f = subfiles(current_folder, join=False)
    f = [i for i in f if i.endswith(suffix)]
    if current_folder.find("fold0") != -1:
        for fl in f:
            shutil.copy(os.path.join(current_folder, fl), os.path.join(out_folder, prefix+fl))
    for su in s:
        if prefix == "":
            add = su
        else:
            add = "__" + su
        recur_copy(os.path.join(current_folder, su), out_folder, prefix=prefix+add)

if __name__ == "__main__":
    from default_configs import net_training_out_dir
    output_folder = "/home/simon/med/results/UNetV2/leaderboard"
    recur_copy(net_training_out_dir, output_folder)
    run_in_folder(output_folder)
