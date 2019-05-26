
from utils.exp_utils import store_seg_from_softmax
from utils.files_utils import *
import numpy as np
from multiprocessing import Pool


def merge_files(args):
    files, properties_file, out_file, store_largest_connected_component, min_region_size_per_class, override = args
    if override or not isfile(out_file):
        softmax = [np.load(f)['softmax'][None] for f in files]
        softmax = np.vstack(softmax)
        softmax = np.mean(softmax, 0)
        props = load_pickle(properties_file)
        store_seg_from_softmax(softmax, out_file, props, 1, None, None, None)


def merge(folders, output_folder, threads, override=True):
    maybe_mkdir_p(output_folder)

    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        assert all([isfile(join(f, i + ".npz")) for i in patient_ids]), "Not all patient npz are available in " \
                                                                        "all folders"
        assert all([isfile(join(f, i + ".pkl")) for i in patient_ids]), "Not all patient pkl are available in " \
                                                                        "all folders"

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        files.append([join(f, p + ".npz") for f in folders])
        property_files.append(join(folders[0], p + ".pkl"))
        out_files.append(join(output_folder, p + ".nii.gz"))

    plans = load_pickle(join(folders[0], "plans.pkl"))

    store_largest_connected_component, min_region_size_per_class = plans['keep_only_largest_region'], \
                                                                       plans['min_region_size_per_class']
    p = Pool(threads)
    p.map(merge_files, zip(files, property_files, out_files, [store_largest_connected_component] * len(out_files),
                           [min_region_size_per_class] * len(out_files), [override] * len(out_files)))
    p.close()
    p.join()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="This requires that all folders to be merged use the same "
                                                 "postprocessing function "
                                                 "(postprocessing.postprocess_segmentation). ")
    parser.add_argument('-f', '--folders', nargs='+', help="list of folders to merge. All folders must contain npz "
                                                           "files", required=True)
    parser.add_argument('-o', '--output_folder', help="where to save the results", required=True, type=str)
    parser.add_argument('-t', '--threads', help="number of threads used to saving niftis", required=False, default=2,
                        type=int)

    args = parser.parse_args()

    folders = args.folders
    threads = args.threads
    output_folder = args.output_folder

    merge(folders, output_folder, threads, override=True)    

if __name__ == "__main__":
    main()
