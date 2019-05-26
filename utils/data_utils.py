#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
#from multiprocessing import Pool

def get_class_balanced_patients(class_targets, batch_size, num_classes, slack_factor=0.1):
    '''
    :param class_targets: list of patient targets. 
    :param batch_size:
    :param num_classes:
    :param slack_factor:
    :return: batch_ixs: list of indices referring to a subset in class_targets-list
    '''
    batch_ixs = []
    class_count = {k: 0 for k in range(num_classes)}
    weakest_class = 0
    for ix in range(batch_size):

        keep_looking = True
        while keep_looking:
            #choose a random patient.
            cand = np.random.choice(len(class_targets), 1)[0]
            # check the least occuring class among this patient's rois.
            tmp_weakest_class = np.argmin([class_targets[cand].count(ii) for ii in range(num_classes)])
            # if current batch already bigger than the slack_factor ratio, then
            # check that weakest class in this patient is not the weakest in current batch (since needs to be boosted)
            # also that at least one roi of this patient belongs to weakest class. If True, keep patient, else keep looking.
            if (tmp_weakest_class != weakest_class and class_targets[cand].count(weakest_class) > 0) or ix < int(batch_size * slack_factor):
                keep_looking = False

        for c in range(num_classes):
            class_count[c] += class_targets[cand].count(c)
        weakest_class = np.argmin(([class_count[c] for c in range(num_classes)]))
        batch_ixs.append(cand)

    return batch_ixs

class fold_generator:

    def __init__(self, seed, n_splits, len_data):
        """
        :param seed: Random seed for splits.
        :param n_splits: number of splits, e.g. 5 splits for 5-fold cross-validation
        :param len_data: number of elements in the dataset.
        """
        self.tr_ix = []
        self.val_ix = []
        self.te_ix = []
        self.slicer = None
        self.missing = 0
        self.fold = 0
        self.len_data = len_data
        self.n_splits = n_splits
        self.myseed = seed
        self.boost_val = 0

    def init_indices(self):

        t = list(np.arange(self.l))
        # round up to next splittable data amount.
        split_length = int(np.ceil(len(t) / float(self.n_splits)))
        self.slicer = split_length
        self.mod = len(t) % self.n_splits
        if self.mod > 0:
            # missing is the number of folds, in which the new splits are reduced to account for missing data.
            self.missing = self.n_splits - self.mod

        self.te_ix = t[:self.slicer]
        self.tr_ix = t[self.slicer:]
        self.val_ix = self.tr_ix[:self.slicer]
        self.tr_ix = self.tr_ix[self.slicer:]

    def new_fold(self):

        slicer = self.slicer
        if self.fold < self.missing :
            slicer = self.slicer - 1

        temp = self.te_ix

        # catch exception mod == 1: test set collects 1+ data since walk through both roudned up splits.
        # account for by reducing last fold split by 1.
        if self.fold == self.n_splits-2 and self.mod ==1:
            temp += self.val_ix[-1:]
            self.val_ix = self.val_ix[:-1]

        self.te_ix = self.val_ix
        self.val_ix = self.tr_ix[:slicer]
        self.tr_ix = self.tr_ix[slicer:] + temp


    def get_fold_names(self):
        names_list = []
        rgen = np.random.RandomState(self.myseed)
        cv_names = np.arange(self.len_data)

        rgen.shuffle(cv_names)
        self.l = len(cv_names)
        self.init_indices()

        for split in range(self.n_splits):
            train_names, val_names, test_names = cv_names[self.tr_ix], cv_names[self.val_ix], cv_names[self.te_ix]
            names_list.append([train_names, val_names, test_names, self.fold])
            self.new_fold()
            self.fold += 1

        return names_list

def get_patch_crop_coords(img, patch_size, min_overlap=64):
    """
    :param img (y, x, (z))
    :param patch_size: list of len 2 (2D) or 3 (3D).
    :param min_overlap: minimum required overlap of patches.
    :return ndarray: shape (n_patches, 2*dim). crop coordinates for each patch.
    """
    crop_coords = []
    for dim in range(len(img.shape)):
        n_patches = int(np.ceil(img.shape[dim] / patch_size[dim]))

        # no crops required in this dimension, add image shape as coordinates.
        if n_patches == 1:
            crop_coords.append([(0, img.shape[dim])])
            continue

        # fix the two outside patches to coords patchsize/2 and interpolate.
        center_dists = (img.shape[dim] - patch_size[dim]) / (n_patches - 1)

        if (patch_size[dim] - center_dists) < min_overlap:
            n_patches += 1
            center_dists = (img.shape[dim] - patch_size[dim]) / (n_patches - 1)

        patch_centers = np.round([(patch_size[dim] / 2 + (center_dists * ii)) for ii in range(n_patches)])
        dim_crop_coords = [(center - patch_size[dim] / 2, center + patch_size[dim] / 2) for center in patch_centers]
        crop_coords.append(dim_crop_coords)

    coords_mesh_grid = []
    for ymin, ymax in crop_coords[0]:
        for xmin, xmax in crop_coords[1]:
            if len(crop_coords) == 3 and patch_size[2] > 1:
                for zmin, zmax in crop_coords[2]:
                    coords_mesh_grid.append([ymin, ymax, xmin, xmax, zmin, zmax])
            elif len(crop_coords) == 3 and patch_size[2] == 1:
                for zmin in range(img.shape[2]):
                    coords_mesh_grid.append([ymin, ymax, xmin, xmax, zmin, zmin + 1])
            else:
                coords_mesh_grid.append([ymin, ymax, xmin, xmax])
    return np.array(coords_mesh_grid).astype(int)

def pad_nd_img(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    :param image: nd image
    :param new_shape: new_shape can be different dimensionality of image.
    :param mode: 
    :param return_slicer: return coords will need to use when cropping back to original shape
    :param shape_must_be_divisible_by:  the new shape is divisibly 
    :param kwargs: 
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer



#############################
#  data packing / unpacking #
#############################

def get_patientIDs(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
    return case_identifiers


def convert_to_npy(npz_file):
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)['data']
        np.save(npz_file[:-3] + "npy", a)


def unpack_dataset(folder, threads=8):
    case_identifiers = get_patientIDs(folder)
    p = Pool(threads)
    npz_files = [os.path.join(folder, i + ".npz") for i in case_identifiers]
    p.map(convert_to_npy, npz_files)
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_patientIDs(folder)
    npy_files = [os.path.join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if os.path.isfile(i)]
    for n in npy_files:
        os.remove(n)


def get_mirrored_patch_crops(patch_crops, org_img_shape):
    """
    patch crop on 3 mirrror transformations (x-axis, y-axis, x&y-axis)
    return the transformed coordinates.
    :param patch_crops: list of crops: a list of coordinates for given crop [[y1, x1, ...], [y1, x1, ..]]
    :param org_img_shape: shape of patient volume used as world coordinates.
    :return: list of mirrored patch crops: lenght=3. each element is a list of transformed patch crops.
    """
    mirrored_patch_crops = []

    # y-axis transform.
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  ii[2], ii[3]] if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  ii[2], ii[3], ii[4], ii[5]] for ii in patch_crops])

    # x-axis transform.
    mirrored_patch_crops.append([[ii[0], ii[1],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]] if len(ii) == 4 else
                                 [ii[0], ii[1],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2],
                                  ii[4], ii[5]] for ii in patch_crops])

    # y-axis and x-axis transform.
    mirrored_patch_crops.append([[org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2]] if len(ii) == 4 else
                                 [org_img_shape[2] - ii[1],
                                  org_img_shape[2] - ii[0],
                                  org_img_shape[3] - ii[3],
                                  org_img_shape[3] - ii[2],
                                  ii[4], ii[5]] for ii in patch_crops])

    return mirrored_patch_crops


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def mean_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.mean(int(ax))
    return inp


def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]
