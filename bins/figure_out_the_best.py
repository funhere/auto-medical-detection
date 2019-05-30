
from utils.files_utils import *
from default_configs import net_training_out_dir
import numpy as np
from evaluation.results_collector import classes_dice_mean
from subprocess import call
import SimpleITK as sitk
from run.default_configuration import get_output_folder


def convert_nifti_to_uint8(args):
    source_file, target_file = args
    i = sitk.ReadImage(source_file)
    j = sitk.GetImageFromArray(sitk.GetArrayFromImage(i).astype(np.uint8))
    j.CopyInformation(i)
    sitk.WriteImage(j, target_file)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(usage="To identify the best model based on the five fold cross-validation."
                                           "Running this script requires all models to have been run already."
                                           "It will summarize the results of the five folds of all "
                                           "models in one json each for easy interpretability")
    parser.add_argument("-m", '--models', nargs="+", required=False, default=['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'])
    parser.add_argument("-t", '--task_ids', nargs="+", required=False, default=list(range(100)))

    args = parser.parse_args()
    tasks = args.task_id
    models = args.models

    out_dir_all_json = join(net_training_out_dir, "summary_jsons")

    json_files = [i for i in subfiles(out_dir_all_json, suffix=".json", join=True) if i.find("ensemble") == -1]

    # do mean over foreground
    for j in json_files:
        classes_dice_mean(j)

    # for each task, run ensembling using all combinations of two models
    for t in tasks:
        json_files_task = [i for i in subfiles(out_dir_all_json, prefix="Task%02.0d_" % t) if i.find("ensemble") == -1]
        if len(json_files_task) > 0:
            task_name = json_files_task[0].split("/")[-1].split("__")[0]
            print(task_name)

            for i in range(len(json_files_task) - 1):
                for j in range(i+1, len(json_files_task)):
                    # nets are stored as
                    # task__configuration__trainer__plans
                    net1 = json_files_task[i].split("/")[-1].split("__")
                    net1[-1] = net1[-1].split(".")[0]
                    task, configuration, trainer, plans_identifier = net1
                    net1_folder = get_output_folder(configuration, task, trainer, plans_identifier)
                    name1 = configuration + "__" + trainer + "__" + plans_identifier

                    net2 = json_files_task[j].split("/")[-1].split("__")
                    net2[-1] = net2[-1].split(".")[0]
                    task, configuration, trainer, plans_identifier = net2
                    net2_folder = get_output_folder(configuration, task, trainer, plans_identifier)
                    name2 = configuration + "__" + trainer + "__" + plans_identifier

                    if np.argsort((name1, name2))[0] == 1:
                        name1, name2 = name2, name1
                        net1_folder, net2_folder = net2_folder, net1_folder

                    output_folder = join(net_training_out_dir, "ensembles", task_name, "ensemble_" + name1 + "--" + name2)
                    # now ensemble
                    print(net1_folder, net2_folder)
                    p = call(["python", join(__path__[0], "evaluation/model_selection/ensemble.py"), net1_folder, net2_folder, output_folder, task_name])

    # rerun adding the mean foreground dice
    json_files = subfiles(out_dir_all_json, suffix=".json", join=True)

    # do mean over foreground
    for j in json_files:
        classes_dice_mean(j)

    # load all json for each task and find best
    with open(join(net_training_out_dir, "use_this_for_test.csv"), 'w') as f:
        for t in tasks:
            json_files_task = subfiles(out_dir_all_json, prefix="Task%02.0d_" % t)
            if len(json_files_task) > 0:
                task_name = json_files_task[0].split("/")[-1].split("__")[0]
                print(task_name)
                mean_dice = []
                for j in json_files_task:
                    js = load_json(j)
                    mean_dice.append(js['results']['mean']['mean']['Dice'])
                best = np.argsort(mean_dice)[::-1][0]
                j = json_files_task[best].split("/")[-1]

                print("%s: submit model %s" % (task_name, j))
                f.write("%s,%s\n" % (task_name, j))
