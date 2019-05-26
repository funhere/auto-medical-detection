
from collections import OrderedDict
import SimpleITK as sitk
from utils.files_utils import *


def export_for_submission(source_dir, target_dir):
    """
    promise wants mhd :-/
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    target_files = [join(target_dir, i[:-7] + ".mhd") for i in files]
    maybe_mkdir_p(target_dir)
    for f, t in zip(files, target_files):
        img = sitk.ReadImage(join(source_dir, f))
        sitk.WriteImage(img, t)


if __name__ == "__main__":
    folder = "/media/simon/med/datasets/promise2012"
    out_folder = "/media/simon/med/MedicalDecathlon/MedicalDecathlon_raw_splitted/Task24_Promise"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    # train
    current_dir = join(folder, "train")
    segmentations = subfiles(current_dir, suffix="segmentation.mhd")
    raw_data = [i for i in subfiles(current_dir, suffix="mhd") if not i.endswith("segmentation.mhd")]
    for i in raw_data:
        out_fname = join(out_folder, "imagesTr", i.split("/")[-1][:-4] + "_0000.nii.gz")
        sitk.WriteImage(sitk.ReadImage(i), out_fname)
    for i in segmentations:
        out_fname = join(out_folder, "labelsTr", i.split("/")[-1][:-17] + ".nii.gz")
        sitk.WriteImage(sitk.ReadImage(i), out_fname)

    # test
    current_dir = join(folder, "test")
    test_data = subfiles(current_dir, suffix="mhd")
    for i in test_data:
        out_fname = join(out_folder, "imagesTs", i.split("/")[-1][:-4] + "_0000.nii.gz")
        sitk.WriteImage(sitk.ReadImage(i), out_fname)


    json_dict = OrderedDict()
    json_dict['name'] = "PROMISE12"
    json_dict['description'] = "prostate"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "f off, this is private"
    json_dict['licence'] = "touch it and you die"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "prostate"
    }
    json_dict['numTraining'] = len(raw_data)
    json_dict['numTest'] = len(test_data)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-4], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-4]} for i in
                             raw_data]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-4] for i in test_data]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

