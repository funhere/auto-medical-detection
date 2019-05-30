
from collections import OrderedDict
from datasets.data_augmentation.aug_utils import random_cropped_2D_img_batch, pad_nd_img
import numpy as np
from preprocessing.data_loader import DataLoaderBase
from multiprocessing import Pool
from default_configs import preprocessing_output_dir
from utils.files_utils import *

class BatchGenerator3D(DataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        """
        Basic data loader for 3D nets.
        :param final_patch_size: 
        :param batch_size:
        :param num_batches: batches will the data loader produce, None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly
        """
        super(BatchGenerator3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())

        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides

    def get_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def gen_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = []
        seg = []
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_oversample(j):
                force_fg = True
            else:
                force_fg = False

            case_properties.append(self._data[i]['properties'])

            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]

                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key+1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None


            need_to_pad = self.need_to_pad
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size pad on both sides
                if need_to_pad[d] + case_all_data.shape[d+1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d+1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(self._data[i]['properties']['classes'])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = 0
                else:
                    selected_class = np.random.choice(foreground_classes)
                voxels_of_that_class = np.argwhere(case_all_data[-1] == selected_class)


                if len(voxels_of_that_class) != 0:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:

                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]


            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                             valid_bbox_y_lb:valid_bbox_y_ub,
                                             valid_bbox_z_lb:valid_bbox_z_ub]
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                                                     valid_bbox_y_lb:valid_bbox_y_ub,
                                                                     valid_bbox_z_lb:valid_bbox_z_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0,0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0,0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                         'constant', **{'constant_values':-1})
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = np.pad(seg_from_previous_stage, ((0,0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                                 'constant', **{'constant_values': 0})
                case_all_data_segonly = np.concatenate((case_all_data_segonly, seg_from_previous_stage), 0)

            data.append(case_all_data_donly[None])
            seg.append(case_all_data_segonly[None])

        data = np.vstack(data)
        seg = np.vstack(seg)

        return {'data':data, 'seg':seg, 'properties':case_properties, 'keys': selected_keys}


class BatchGenerator2D(DataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, transpose=None,
                 oversample_foreground_percent=0.0, memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D nets. 
        :param data: get this with load_dataset(folder, stage=0). 
        :param patch_size: patch size will this data loader return.
        :param final_patch_size: patch finally be cropped
        :param batch_size:
        :param num_batches: batches will the data loader produce, None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly
        """
        super(BatchGenerator2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        if transpose is not None:
            assert isinstance(transpose, (list, tuple)), "Transpose must be either None or be a tuple/list representing the new axis order (3 ints)"
        self.transpose = transpose
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides

    @property
    def all_slices(self):
        try:
            return self._all_slices
        except AttributeError:
            slices = []
            for key in sorted(self.list_of_keys):
                shape = self._data[key]["properties"]["size_after_resampling"]
                for s in range(shape[0]):
                    slices.append((key, s))
            self._all_slices = slices
            return slices

    def get_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def gen_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = []
        seg = []
        case_properties = []
        for j, i in enumerate(selected_keys):
            properties = self._data[i]['properties']
            case_properties.append(properties)

            if self.get_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            if self.transpose is not None:
                leading_axis = self.transpose[0]
            else:
                leading_axis = 0

            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[leading_axis + 1])
            else:
                # select one class, then select a slice that contains that class
                classes_in_slice_per_axis = properties.get("classes_in_slice_per_axis")
                possible_classes = np.unique(properties['classes'])
                possible_classes = possible_classes[possible_classes > 0]
                if len(possible_classes) > 0 and not (0 in possible_classes.shape):
                    selected_class = np.random.choice(possible_classes)
                else:
                    selected_class = 0
                if classes_in_slice_per_axis is not None:
                    valid_slices = classes_in_slice_per_axis[leading_axis][selected_class]
                else:
                    valid_slices = np.where(np.sum(case_all_data[-1] == selected_class, axis=[i for i in range(3) if i != leading_axis]))[0]
                if len(valid_slices) != 0:
                    random_slice = np.random.choice(valid_slices)
                else:
                    random_slice = np.random.choice(case_all_data.shape[leading_axis + 1])

            if self.pseudo_3d_slices == 1:
                if leading_axis == 0:
                    case_all_data = case_all_data[:, random_slice]
                elif leading_axis == 1:
                    case_all_data = case_all_data[:, :, random_slice]
                else:
                    case_all_data = case_all_data[:, :, :, random_slice]
                if self.transpose is not None and self.transpose[1] > self.transpose[2]:
                    case_all_data = case_all_data.transpose(0, 2, 1)
            else:
                assert leading_axis == 0, "pseudo_3d_slices works only without transpose for now!"
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            num_seg = 1

            new_shp = None
            if np.any(self.need_to_pad) > 0:
                new_shp = np.array(case_all_data.shape[1:] + self.need_to_pad)
                if np.any(new_shp - np.array(self.patch_size) < 0):
                    new_shp = np.max(np.vstack((new_shp[None], np.array(self.patch_size)[None])), 0)
            else:
                if np.any(np.array(case_all_data.shape[1:]) - np.array(self.patch_size) < 0):
                    new_shp = np.max(
                        np.vstack((np.array(case_all_data.shape[1:])[None], np.array(self.patch_size)[None])), 0)
            if new_shp is not None:
                case_all_data_donly = pad_nd_img(case_all_data[:-num_seg], new_shp, self.pad_mode, kwargs=self.pad_kwargs_data)
                case_all_data_segnonly = pad_nd_img(case_all_data[-num_seg:], new_shp, 'constant', kwargs={'constant_values':-1})
                case_all_data = np.vstack((case_all_data_donly, case_all_data_segnonly))[None]
            else:
                case_all_data = case_all_data[None]

            if not force_fg:
                case_all_data = random_cropped_2D_img_batch(case_all_data, tuple(self.patch_size))
            else:
                case_all_data = crop_2D_img(case_all_data[0], tuple(self.patch_size), selected_class)[None]
            data.append(case_all_data[:, :-num_seg])
            seg.append(case_all_data[:, -num_seg:])
        data = np.vstack(data)
        seg = np.vstack(seg)
        keys = selected_keys
        return {'data':data, 'seg':seg, 'properties':case_properties, "keys": keys}

def get_patientIDs(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def get_patientIDs_from_raw_folder(folder):
    case_identifiers = np.unique([i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz") and (i.find("segFromPrevStage") == -1)])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=8, key="data"):
    """
    unpacks all npz files in a folder to npy
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key]*len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=8, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key]*len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_patientIDs(folder)
    npy_files = [join(folder, i+".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder):
    case_identifiers = get_patientIDs(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz"%c)
        with open(join(folder, "%s.pkl"%c), 'rb') as f:
            dataset[c]['properties'] = pickle.load(f)
        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz"%c)
    return dataset


def crop_2D_img(img, crop_size, force_class=None):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :return:
    """
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If center crop is a list/tuple, make sure it has the same length as data has dims (3d)"

    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    foreground_classes = np.unique(img[-1])
    foreground_classes = foreground_classes[foreground_classes > 0]
    if len(foreground_classes) == 0 or (0 in foreground_classes.shape):
        foreground_classes = [0]

    if force_class is None or force_class not in foreground_classes:
        chosen_class = np.random.choice(foreground_classes)
    else:
        chosen_class = force_class
    foreground_voxels = np.array(np.where(img[-1] == chosen_class))
    if np.any(np.array(foreground_voxels.shape) == 0):
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = foreground_voxels[:, np.random.choice(foreground_voxels.shape[1])]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i]//2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i+1] - crop_size[i]//2 - crop_size[i] % 2, selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0]//2):(selected_center_voxel[0] + crop_size[0]//2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1]//2):(selected_center_voxel[1] + crop_size[1]//2 + crop_size[1] % 2)]
    return result



if __name__ == "__main__":
    t = "Task_Heart"
    p = join(preprocessing_output_dir, t, "stage1")
    dataset = load_dataset(p)
    with open(join(join(preprocessing_output_dir, t), "plans_stage1.pkl"), 'rb') as f:
        plans = pickle.load(f)
    unpack_dataset(p)
    dl = BatchGenerator3D(dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33)
    dl = BatchGenerator3D(dataset, np.array(plans['patch_size']).astype(int), np.array(plans['patch_size']).astype(int), 2, oversample_foreground_percent=0.33)
    dl2d = BatchGenerator2D(dataset, (64, 64), np.array(plans['patch_size']).astype(int)[1:], 12, oversample_foreground_percent=0.33)
