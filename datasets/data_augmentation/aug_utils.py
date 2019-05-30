
from __future__ import print_function
from builtins import range, zip
import random
import numpy as np
from copy import deepcopy
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from skimage.transform import resize
from scipy.ndimage.measurements import label as lb

def gen_elastic_transform_coord(shape, alpha, sigma):
    n_dim = len(shape)
    offsets = []
    for _ in range(n_dim):
        offsets.append(gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.meshgrid(*tmp, indexing='ij')
    indices = [np.reshape(i + j, (-1, 1)) for i, j in zip(offsets, coords)]
    return indices


def gen_zero_centered_coord_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def convert_seg_to_one_hot_encoding(image, classes=None):
    '''
    Input: an nd array of a label map (any dimension). 
    Outputs: a one hot encoding of the label map.
    Example (3D): if input is of shape (x, y, z), the output will ne of shape (n_classes, x, y, z)
    '''
    if classes is None:
        classes = np.unique(image)
    out_image = np.zeros([len(classes)]+list(image.shape), dtype=image.dtype)
    for i, c in enumerate(classes):
        out_image[i][image == c] = 1
    return out_image


def elastic_deform_coords(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def rotate_3D_coords(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = gen_matrix_rotation_x_3D(angle_x, rot_matrix)
    rot_matrix = gen_matrix_rotation_y_3D(angle_y, rot_matrix)
    rot_matrix = gen_matrix_rotation_z_3D(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def rotate_2D_coords(coords, angle):
    rot_matrix = gen_matrix_rotation_2D(angle)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def scale_coords(coords, scale):
    return coords * scale


def uncenter_coords(coords):
    shp = coords.shape[1:]
    coords = deepcopy(coords)
    for d in range(coords.shape[0]):
        coords[d] += (shp[d] - 1) / 2.
    return coords


def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)


def gen_noise(shape, alpha, sigma):
    noise = np.random.random(shape) * 2 - 1
    noise = gaussian_filter(noise, sigma, mode="constant", cval=0) * alpha
    return noise


def get_entries_in_array(entries, myarray):
    entries = np.array(entries)
    values = np.arange(np.max(myarray) + 1)
    lut = np.zeros(len(values), 'bool')
    lut[entries.astype("int")] = True
    return np.take(lut, myarray.astype(int))


def center_cropped_3D_img(img, crop_size):
    center = np.array(img.shape) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
           int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]


def center_cropped_3D_img_batch(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2, 3 and 4 are x y z
    center = np.array(img.shape[2:]) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * (len(img.shape) - 2)
    else:
        center_crop = crop_size
        assert len(center_crop) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
           int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]


def center_cropped_2D_img(img, crop_size):
    center = np.array(img.shape) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]


def center_cropped_2D_img_batched(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2 and 3 are x y
    center = np.array(img.shape[2:]) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * (len(img.shape) - 2)
    else:
        center_crop = crop_size
        assert len(center_crop) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]


def random_cropped_3D_img(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[2]:
        lb_z = np.random.randint(0, img.shape[2] - crop_size[2])
    elif crop_size[2] == img.shape[2]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]


def random_cropped_3D_img_batch(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")

    return img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]


def random_cropped_2D_img(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]


def random_cropped_2D_img_batch(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    return img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]


def resize_img_by_padding(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res


def resize_img_by_padding_batch(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape[2:]))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res = np.ones((image.shape[0], image.shape[1], new_shape[0], new_shape[1]), dtype=image.dtype) * pad_value
        res[:, :, int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image[:,
                                                                                                              :]
    elif len(shape) == 3:
        res = np.ones((image.shape[0], image.shape[1], new_shape[0], new_shape[1], new_shape[2]),
                      dtype=image.dtype) * pad_value
        res[:, :, int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image[:, :]
    else:
        raise RuntimeError("unexpected dimension")
    return res


def gen_matrix_rotation_x_3D(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def gen_matrix_rotation_y_3D(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def gen_matrix_rotation_z_3D(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def gen_matrix_rotation_2D(angle, matrix=None):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)


def gen_random_rotation(angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi)):
    return gen_matrix_rotation_x_3D(np.random.uniform(*angle_x),
                                       gen_matrix_rotation_y_3D(np.random.uniform(*angle_y),
                                                                   gen_matrix_rotation_z_3D(
                                                                       np.random.uniform(*angle_z))))


def imp_jitter(img, u, s, sigma):
    # img must have shape [....., c] where c is the color channel
    alpha = np.random.normal(0, sigma, s.shape)
    jitter = np.dot(u, alpha * s)
    img2 = np.array(img)
    for c in range(img.shape[0]):
        img2[c] = img[c] + jitter[c]
    return img2


def gen_cc_var_num_channels(img, diff_order=0, mink_norm=1, sigma=1, mask_im=None, saturation_threshold=255,
                                dilation_size=3, clip_range=True):
    # img must have first dim color channel! img[c, x, y(, z, ...)]
    dim_img = len(img.shape[1:])
    if clip_range:
        minm = img.min()
        maxm = img.max()
    img_internal = np.array(img)
    if mask_im is None:
        mask_im = np.zeros(img_internal.shape[1:], dtype=bool)
    img_dil = deepcopy(img_internal)
    for c in range(img.shape[0]):
        img_dil[c] = grey_dilation(img_internal[c], tuple([dilation_size] * dim_img))
    mask_im = mask_im | np.any(img_dil >= saturation_threshold, axis=0)
    if sigma != 0:
        mask_im[:sigma, :] = 1
        mask_im[mask_im.shape[0] - sigma:, :] = 1
        mask_im[:, mask_im.shape[1] - sigma:] = 1
        mask_im[:, :sigma] = 1
        if dim_img == 3:
            mask_im[:, :, mask_im.shape[2] - sigma:] = 1
            mask_im[:, :, :sigma] = 1

    out_img = deepcopy(img_internal)

    if diff_order == 0 and sigma != 0:
        for c in range(img_internal.shape[0]):
            img_internal[c] = gaussian_filter(img_internal[c], sigma, diff_order)
    elif diff_order == 1:
        for c in range(img_internal.shape[0]):
            img_internal[c] = gaussian_gradient_magnitude(img_internal[c], sigma)
    elif diff_order > 1:
        raise ValueError("diff_order can only be 0 or 1. 2 is not supported (ToDo, maybe)")

    img_internal = np.abs(img_internal)

    white_colors = []

    if mink_norm != -1:
        kleur = np.power(img_internal, mink_norm)
        for c in range(kleur.shape[0]):
            white_colors.append(np.power((kleur[c][mask_im != 1]).sum(), 1. / mink_norm))
    else:
        for c in range(img_internal.shape[0]):
            white_colors.append(np.max(img_internal[c][mask_im != 1]))

    som = np.sqrt(np.sum([i ** 2 for i in white_colors]))

    white_colors = [i / som for i in white_colors]

    for c in range(out_img.shape[0]):
        out_img[c] /= (white_colors[c] * np.sqrt(3.))

    if clip_range:
        out_img[out_img < minm] = minm
        out_img[out_img > maxm] = maxm
    return white_colors, out_img


def convert_seg_to_bbox_coords(data_dict, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):

        '''
        :param data_dict:
        :param dim:
        :param get_rois_from_seg:
        :return: coords (y1, x1, y2, x2)
        '''

        bb_target = []
        roi_masks = []
        roi_labels = []
        out_seg = np.copy(data_dict['seg'])
        for b in range(data_dict['seg'].shape[0]):

            p_coords_list = []
            p_roi_masks_list = []
            p_roi_labels_list = []

            if np.sum(data_dict['seg'][b]!=0) > 0:
                if get_rois_from_seg_flag:
                    clusters, n_cands = lb(data_dict['seg'][b])
                    data_dict['class_target'][b] = [data_dict['class_target'][b]] * n_cands
                else:
                    n_cands = int(np.max(data_dict['seg'][b]))
                    clusters = data_dict['seg'][b]

                rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
                for rix, r in enumerate(rois):
                    if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
                        seg_ixs = np.argwhere(r != 0)
                        coord_list = [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                         np.max(seg_ixs[:, 2])+1]
                        if dim == 3:

                            coord_list.extend([np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1])

                        p_coords_list.append(coord_list)
                        p_roi_masks_list.append(r)
                        # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                        # also patient wide, this assignment is not dependent on patch occurrances.
                        p_roi_labels_list.append(data_dict['class_target'][b][rix] + 1)

                    if class_specific_seg_flag:
                        out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_target'][b][rix] + 1

                if not class_specific_seg_flag:
                    out_seg[b][data_dict['seg'][b] > 0] = 1

                bb_target.append(np.array(p_coords_list))
                roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
                roi_labels.append(np.array(p_roi_labels_list))


            else:
                bb_target.append([])
                roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
                roi_labels.append(np.array([-1]))

        if get_rois_from_seg_flag:
            data_dict.pop('class_target', None)

        data_dict['bb_target'] = np.array(bb_target)
        data_dict['roi_masks'] = np.array(roi_masks)
        data_dict['roi_labels'] = np.array(roi_labels)
        data_dict['seg'] = out_seg

        return data_dict


def trans_channels(batch):
    if len(batch.shape) == 4:
        return np.transpose(batch, axes=[0, 2, 3, 1])
    elif len(batch.shape) == 5:
        return np.transpose(batch, axes=[0, 4, 2, 3, 1])
    else:
        raise ValueError("wrong dimensions in transpose_channel generator!")


def resize_seg(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resize_multichannel_img(multichannel_image, new_shape, order=3):
    '''
    Resizes multichannel_image. Resizes each channel in c separately and fuses results back together

    :param multichannel_image: c x x x y (x z)
    :param new_shape: x x y (x z)
    :param order:
    :return:
    '''
    tpe = multichannel_image.dtype
    new_shp = [multichannel_image.shape[0]] + list(new_shape)
    result = np.zeros(new_shp, dtype=multichannel_image.dtype)
    for i in range(multichannel_image.shape[0]):
        result[i] = resize(multichannel_image[i].astype(float), new_shape, order, "constant", 0, True, anti_aliasing=False)
    return result.astype(tpe)


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


def uniform(low, high, size=None):
    """
    wrapper for np.random.uniform to allow it to handle low=high
    :param low:
    :param high:
    :return:
    """
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)


def pad_nd_img(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit
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



def random_mask_square(img, square_size, n_val, channel_wise_n_val=False, square_pos=None):
    """Masks (sets = 0) a random square in an image"""

    img_h = img.shape[-2]
    img_w = img.shape[-1]

    img = img.copy()

    if square_pos is None:
        w_start = np.random.randint(0, img_w - square_size)
        h_start = np.random.randint(0, img_h - square_size)
    else:
        pos_wh = square_pos[np.random.randint(0, len(square_pos))]
        w_start = pos_wh[0]
        h_start = pos_wh[1]

    if img.ndim == 2:
        rnd_n_val = get_range_val(n_val)
        img[h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
    elif img.ndim == 3:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[:, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
    elif img.ndim == 4:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[:, i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[:, :, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val

    return img


def random_mask_squares(img, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
    """Masks a given number of squares in an image"""
    for i in range(n_squares):
        img = random_mask_square(img, square_size, n_val, channel_wise_n_val=channel_wise_n_val,
                                 square_pos=square_pos)
    return img


def convert_seg_to_bbox_coords(data_dict, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):
    """Converts a mask into a bounding box format.
    data: format (n_patches, c, x, y, z)
    mask: format (n_patches, 1, x, y, z)
    """

    bb_target = []
    roi_masks = []
    roi_labels = []
    out_seg = np.copy(data_dict['seg'])
    for bix in range(data_dict['seg'].shape[0]):

        p_coords_list = []
        p_roi_masks_list = []
        p_roi_labels_list = []

        if np.sum(data_dict['seg'][bix]!=0) > 0:
            if get_rois_from_seg_flag:
                clusters, n_cands = lb(data_dict['seg'][bix])
                #s = generate_binary_structure(2,2) #(1,1); (3,3)
                #clusters, n_cands = lb(data_dict['seg'][bix], structure=s)
                #data_dict['class_target'][bix] = [data_dict['class_target'][bix]] * n_cands
            else:
                n_cands = int(np.max(data_dict['seg'][bix]))
                clusters = data_dict['seg'][bix]

            rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
            for rix, r in enumerate(rois):
                if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
                    seg_ixs = np.argwhere(r != 0)
                    coord_list = [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                     np.max(seg_ixs[:, 2])+1]
                    if dim == 3:

                        coord_list.extend([np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1])

                    p_coords_list.append(coord_list)
                    p_roi_masks_list.append(r)
                    # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                    # also patient wide, this assignment is not dependent on patch occurrances.
                    p_roi_labels_list.append(data_dict['class_target'][bix])#[rix] + 1)

                if class_specific_seg_flag:
                    out_seg[bix][data_dict['seg'][bix] == rix + 1] = data_dict['class_target'][bix][rix] + 1

            if not class_specific_seg_flag:
                out_seg[bix][data_dict['seg'][bix] > 0] = 1

            bb_target.append(np.array(p_coords_list))
            roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
            roi_labels.append(np.array(p_roi_labels_list))

        else:
            bb_target.append([])
            roi_masks.append(np.zeros_like(data_dict['seg'][bix])[None])
            roi_labels.append(np.array([-1]))

    if get_rois_from_seg_flag:
        data_dict.pop('class_target', None)

    data_dict['bb_target'] = np.array(bb_target) #box['box_coords']
    data_dict['roi_masks'] = np.array(roi_masks)
    data_dict['roi_labels'] = np.array(roi_labels)
    data_dict['seg'] = out_seg

    return data_dict
