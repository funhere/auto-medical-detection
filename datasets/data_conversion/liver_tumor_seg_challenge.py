
from collections import OrderedDict
import SimpleITK as sitk
from utils.files_utils import *
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import label


def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    train_dir = "/media/simon/DeepLearningData/tmp/LITS-Challenge-Train-Data"
    test_dir = "/media/simon/med/datasets/LiTS/test_data"


    output_folder = "/media/simon/med/MedicalDecathlon/MedicalDecathlon_raw_splitted/Task_LITS"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "train_" + pat_id.split("-")[-1][:-4]

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))

        img_itk = sitk.ReadImage(seg_file)
        sitk.WriteImage(img_itk, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id

    def load_save_test(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "test_" + pat_id.split("-")[-1][:-4]

        img_itk = sitk.ReadImage(data_file)
        sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
        return pat_id

    nii_files_tr_data = subfiles(train_dir, True, "volume", "nii", True)
    nii_files_tr_seg = subfiles(train_dir, True, "segmen", "nii", True)

    nii_files_ts = subfiles(test_dir, True, "test-volume", "nii", True)

    p = Pool(8)
    train_ids = p.map(load_save_train, zip(nii_files_tr_data, nii_files_tr_seg))
    test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "LITS"
    json_dict['description'] = "LITS"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "f off, this is private"
    json_dict['licence'] = "touch it and you die"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "liver",
        "2": "tumor"
    }

    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)