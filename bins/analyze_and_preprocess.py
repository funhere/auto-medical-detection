
from utils.analysis_utils import contain_classes_in_slice, split_4D_nifti
from preprocessing.cropping import ImgCropper
from utils.files_utils import *
from default_configs import splitted_4D_out_dir, cropped_output_dir, preprocessing_output_dir, raw_dataset_dir
import numpy as np
import pickle
from analysis.DatasetAnalyzer import DatasetAnalyzer
import os
from multiprocessing import Pool
import json
import shutil
from utils.files_utils import *


def split_4D(task_string):
    base_folder = join(raw_dataset_dir, task_string)
    output_folder = join(splitted_4D_out_dir, task_string)

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(base_folder, subdir)
        nii_files = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(base_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(8)
    p.starmap(split_4D_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(base_folder, "dataset.json"), output_folder)


def get_lists_of_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}

def get_caseIDs_of_splitted_dataset(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    files = [i[:-12] for i in files]
    # only unique patient ids
    files = np.unique(files)
    return files


def get_folder_list_of_splitted_dataset(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_of_splitted_dataset(folder)
    list_of_lists = []
    for f in caseIDs:
        list_of_lists.append(subfiles(folder, prefix=f, suffix=".nii.gz", join=True, sort=True))
    return list_of_lists


def crop(task_string, override=False, num_threads=8):
    cropped_out_dir = join(cropped_output_dir, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4D_out_dir_task = join(splitted_4D_out_dir, task_string)
    lists, _ = get_lists_of_splitted_dataset(splitted_4D_out_dir_task)

    imgcrop = ImgCropper(num_threads, cropped_out_dir)
    imgcrop.do_cropping(lists, overwrite_existing=override)
    shutil.copy(join(splitted_4D_out_dir, task_string, "dataset.json"), cropped_out_dir)


def analyze_dataset(task_string, override=False, collect_intensityproperties=True, num_processes=8):
    cropped_out_dir = join(cropped_output_dir, task_string)
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=override, num_processes=num_processes)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)


def plan_and_preprocess(task_string, num_threads=8, no_preprocessing=False):
    from analysis.planner_2D import Planner2D
    from analysis.planner_3D import Planner

    preprocessing_out_dir_train = join(preprocessing_output_dir, task_string)
    cropped_out_dir = join(cropped_output_dir, task_string)
    maybe_mkdir_p(preprocessing_out_dir_train)

    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_out_dir_train)
    shutil.copy(join(splitted_4D_out_dir, task_string, "dataset.json"), preprocessing_out_dir_train)

    exp_planner = Planner(cropped_out_dir, preprocessing_out_dir_train)
    exp_planner.plan_exps()
    if not no_preprocessing:
        exp_planner.do_preprocessing(num_threads)

    exp_planner = Planner2D(cropped_out_dir, preprocessing_out_dir_train)
    exp_planner.plan_exps()
    if not no_preprocessing:
        exp_planner.do_preprocessing(num_threads)

    if not no_preprocessing:
        p = Pool(8)

        stages = [i for i in subdirs(preprocessing_out_dir_train, join=True, sort=True)
                  if i.split("/")[-1].find("stage") != -1]
        for s in stages:
            print(s.split("/")[-1])
            list_of_npz_files = subfiles(s, True, None, ".npz", True)
            list_of_pkl_files = [i[:-4]+".pkl" for i in list_of_npz_files]
            all_classes = []
            for pk in list_of_pkl_files:
                with open(pk, 'rb') as f:
                    props = pickle.load(f)
                all_classes_tmp = np.array(props['classes'])
                all_classes.append(all_classes_tmp[all_classes_tmp >= 0])
            p.map(contain_classes_in_slice, zip(list_of_npz_files, list_of_pkl_files, all_classes))
        p.close()
        p.join()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, 
                        help="task name. There should be a matching folder in "
                                                       "raw_dataset_dir", required=True)
    parser.add_argument('-p', '--processes', type=int, default=3, 
                        help='number of processes to do preprocessing, Default: 3', required=False)
    parser.add_argument('-o', '--override', type=int, default=0, 
                        help="1: override cropped data and intensityproperties. Default: 0",
                        required=False)
    parser.add_argument('-s', '--use_splitted', type=int, default=1, 
                        help='1: use splitted data if already present (skip split_4D).' 
                        '0: do splitting again. Default: 1', required=False)
    parser.add_argument('-no_preprocessing', type=int, default=0, 
                        help='debug only. 1: only run experiment planning, not run preprocessing.')

    args = parser.parse_args()
    task = args.task
    processes = args.processes
    override = args.override
    use_splitted = args.use_splitted
    no_preprocessing = args.no_preprocessing

    if override == 0:
        override = False
    elif override == 1:
        override = True
    else:
        raise ValueError("only 0 or 1 allowed for override")

    if no_preprocessing == 0:
        no_preprocessing = False
    elif no_preprocessing == 1:
        no_preprocessing = True
    else:
        raise ValueError("only 0 or 1 allowed for override")

    if use_splitted == 0:
        use_splitted = False
    elif use_splitted == 1:
        use_splitted = True
    else:
        raise ValueError("only 0 or 1 allowed for use_splitted")

    if task == "all":
        all_tasks_that_need_splitting = subdirs(raw_dataset_dir, prefix="Task", join=False)

        for t in all_tasks_that_need_splitting:
            if not use_splitted or not isdir(join(splitted_4D_out_dir, t)):
                print("splitting task ", t)
                split_4D(t)

        all_splitted_tasks = subdirs(splitted_4D_out_dir, prefix="Task", join=False)
        for t in all_splitted_tasks:
            crop(t, override=override, num_threads=processes)
            analyze_dataset(t, override=override, collect_intensityproperties=True, num_processes=processes)
            plan_and_preprocess(t, processes, no_preprocessing)
    else:
        if not use_splitted or not isdir(join(splitted_4D_out_dir, task)):
            print("splitting task ", task)
            split_4D(task)

        crop(task, override=override, num_threads=processes)
        analyze_dataset(task, override, collect_intensityproperties=True, num_processes=processes)
        plan_and_preprocess(task, processes, no_preprocessing)

if __name__ == "__main__":
    main()
