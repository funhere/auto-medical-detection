from __future__ import print_function
from future import standard_library

import abc
from warnings import warn

import numpy as np
import sys
import logging

import threading
standard_library.install_aliases()
from builtins import range
from multiprocessing import Process
from multiprocessing import Queue
from queue import Queue as thrQueue

from datasets.data_augmentation.aug_utils import gen_zero_centered_coord_mesh, \
    elastic_deform_coords, interpolate_img, rotate_2D_coords, rotate_3D_coords, \
    scale_coords, 
    

class AbstractAugmentation(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class RndTransform(AbstractAugmentation):
    """Applies a augmentation with a specified probability

    Args:
        transform: The augmentation (or composed augmentation)
        prob: The probability with which to apply it
    """

    def __init__(self, transform, prob=0.5, alternative_transform=None):
        self.alternative_transform = alternative_transform
        self.transform = transform
        self.prob = prob

    def __call__(self, **data_dict):
        rnd_val = np.random.uniform()

        if rnd_val < self.prob:
            return self.transform(**data_dict)
        else:
            if self.alternative_transform is not None:
                return self.alternative_transform(**data_dict)
            else:
                return data_dict


class Compose(AbstractAugmentation):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.


    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class RenameTransform(AbstractAugmentation):
    '''
    Saves the value of data_dict[in_key] to data_dict[out_key]. Optionally removes data_dict[in_key] from the dict.
    '''

    def __init__(self, in_key, out_key, delete_old=False):
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, **data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict


class RemoveLabelTransform(AbstractTransform):
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict


class NumpyToTensor(AbstractTransform):
    def __init__(self, keys=None, cast_to=None):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors

        """
        if keys is not None and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.half()
            elif self.cast_to == 'float':
                tensor = tensor.float()
            elif self.cast_to == 'long':
                tensor = tensor.long()
            else:
                raise ValueError('Unknown value for cast_to: %s' % self.cast_to)
        return tensor

    def __call__(self, **data_dict):
        import torch

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(val))
        else:
            for key in self.keys:
                data_dict[key] = self.cast(torch.from_numpy(data_dict[key]))

        return data_dict

#####################################
#     Color Augmenter      #
#####################################
class GammaTransform(AbstractAugmentation):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. 
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range, self.invert_image,
                                                         per_channel=self.per_channel, retain_stats=self.retain_stats)
        return data_dict
    
    def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                      retain_stats=False):
        if invert_image:
            data_sample = - data_sample
        if not per_channel:
            if retain_stats:
                mn = data_sample.mean()
                sd = data_sample.std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample.min()
            rnge = data_sample.max() - minm
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            if retain_stats:
                data_sample = data_sample - data_sample.mean() + mn
                data_sample = data_sample / (data_sample.std() + 1e-8) * sd
        else:
            for c in range(data_sample.shape[0]):
                if retain_stats:
                    mn = data_sample[c].mean()
                    sd = data_sample[c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data_sample[c].min()
                rnge = data_sample[c].max() - minm
                data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                if retain_stats:
                    data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                    data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
        if invert_image:
            data_sample = - data_sample        
        

#####################################
#     Crop Augmenter      #
#####################################
  
      

#####################################
#     Resample Augmenter      #
#####################################
  

#####################################
#     Spatial Augmenter      #
#####################################    
class SpatialTransform(AbstractAugmentation):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: 
    """
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data", label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1):
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample)

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict  
        
    def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=30,
                        do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                        do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                        do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                        p_scale_per_sample=1, p_rot_per_sample=1):
        dim = len(patch_size)
        seg_result = None
        if seg is not None:
            if dim == 2:
                seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
            else:
                seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                      dtype=np.float32)
    
        if dim == 2:
            data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                   dtype=np.float32)
    
        if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
            patch_center_dist_from_border = dim * [patch_center_dist_from_border]
        for sample_id in range(data.shape[0]):
            coords = gen_zero_centered_coord_mesh(patch_size)
            modified_coords = False
            if np.random.uniform() < p_el_per_sample and do_elastic_deform:
                a = np.random.uniform(alpha[0], alpha[1])
                s = np.random.uniform(sigma[0], sigma[1])
                coords = elastic_deform_coords(coords, a, s)
                modified_coords = True
            if np.random.uniform() < p_rot_per_sample and do_rotation:
                if angle_x[0] == angle_x[1]:
                    a_x = angle_x[0]
                else:
                    a_x = np.random.uniform(angle_x[0], angle_x[1])
                if dim == 3:
                    if angle_y[0] == angle_y[1]:
                        a_y = angle_y[0]
                    else:
                        a_y = np.random.uniform(angle_y[0], angle_y[1])
                    if angle_z[0] == angle_z[1]:
                        a_z = angle_z[0]
                    else:
                        a_z = np.random.uniform(angle_z[0], angle_z[1])
                    coords = rotate_3D_coords(coords, a_x, a_y, a_z)
                else:
                    coords = rotate_2D_coords(coords, a_x)
                modified_coords = True
            if np.random.uniform() < p_scale_per_sample and do_scale:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])
                coords = scale_coords(coords, sc)
                modified_coords = True
            # now find a nice center location
            if modified_coords:
                for d in range(dim):
                    if random_crop:
                        ctr = np.random.uniform(patch_center_dist_from_border[d],
                                                data.shape[d + 2] - patch_center_dist_from_border[d])
                    else:
                        ctr = int(np.round(data.shape[d + 2] / 2.))
                    coords[d] += ctr
                for channel_id in range(data.shape[1]):
                    data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                         border_mode_data, cval=border_cval_data)
                if seg is not None:
                    for channel_id in range(seg.shape[1]):
                        seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                            border_mode_seg, cval=border_cval_seg, is_seg=True)
            else:
                if seg is None:
                    s = None
                else:
                    s = seg[sample_id:sample_id + 1]
                if random_crop:
                    margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                    d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
                else:
                    d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
                data_result[sample_id] = d[0]
                if seg is not None:
                    seg_result[sample_id] = s[0]
        return data_result, seg_result


class MirrorTransform(AbstractAugmentation):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. 
    Probability of mirroring along each axis is 0.5
    
    Args:
        axes (tuple of int): axes along which to mirror
    """
    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]
            ret_val = augment_mirroring(data[b], sample_seg, axes=self.axes)
            data[b] = ret_val[0]
            if seg is not None:
                seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
    

#####################################
#     Noise Augmenter      #
##################################### 


#####################################
#     Channel Augmenter      #
##################################### 
class DataChannelSelectionTransform(AbstractTransform):
    """Selects color channels from the batch and discards the others.

    Args:
        channels (list of int): List of channels to be kept.

    """

    def __init__(self, channels, data_key="data"):
        self.data_key = data_key
        self.channels = channels

    def __call__(self, **data_dict):
        data_dict[self.data_key] = data_dict[self.data_key][:, self.channels]
        return data_dict


class SegChannelSelectionTransform(AbstractTransform):
    """Segmentations may have more than one channel. This transform selects segmentation channels
    Args:
        channels (list of int): List of channels to be kept.
    """

    def __init__(self, channels, keep_discarded_seg=False, label_key="seg"):
        self.label_key = label_key
        self.channels = channels
        self.keep_discarded = keep_discarded_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn("No 'seg' key in data_dict while SegChannelSelectionTransform, returning data_dict unmodified", Warning)
        else:
            if self.keep_discarded:
                discarded_seg_idx = [i for i in range(len(seg[0])) if i not in self.channels]
                data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
            data_dict[self.label_key] = seg[:, self.channels]
        return data_dict
        
    
    

#####################################
#     Multi-Threaded Augmenter      #
#####################################

def producer(queue, data_loader, transform, thread_id, seed):
    try:
        np.random.seed(seed)
        data_loader.set_thread_id(thread_id)
        while True:
            for item in data_loader:
                if transform is not None:
                    item = transform(**item)
                queue.put(item)
            queue.put("end")
    except KeyboardInterrupt:
        queue.put("end")
        raise KeyboardInterrupt


def pin_memory_loop(in_queues, out_queue):
    import torch
    queue_ctr = 0
    while True:
        item = in_queues[queue_ctr % len(in_queues)].get()
        if isinstance(item, dict):
            for k in item.keys():
                if isinstance(item[k], torch.Tensor):
                    item[k] = item[k].pin_memory()
        queue_ctr += 1
        out_queue.put(item)
        
        
class MultiThreadedAugmenter(object):
    """ Makes pipeline multi threaded. 

    Args:
     :data_loader (generator or DataLoaderBase instance): data loader. Must have a .next() function and return
        a dict that complies with data structure
     :transform (Transform instance): Any of transformations. 
     :num_processes (int): number of processes
     :num_cached_per_queue (int): number of batches cached per process. 2 to be ideal.
     :seeds (list of int): one seed for each worker. Must have length.If None then seeds = range(num_processes)
     pin_memory (bool): set to True if all torch tensors in data_dict are to be pinned. Pytorch only.
    """
    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False):
        self.pin_memory = pin_memory
        self.transform = transform
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = list(range(num_processes))
        self.seeds = seeds
        self.generator = data_loader
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self._queues = []
        self._threads = []
        self._end_ctr = 0
        self._queue_loop = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None

    def __iter__(self):
        return self

    def _next_queue(self):
        r = self._queue_loop
        self._queue_loop += 1
        if self._queue_loop == self.num_processes:
            self._queue_loop = 0
        return r

    def __next__(self):
        if len(self._queues) == 0:
            self._start()
        try:
            if not self.pin_memory:
                item = self._queues[self._next_queue()].get()
            else:
                item = self.pin_memory_queue.get()

            while item == "end":
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    self._end_ctr = 0
                    self._queue_loop = 0
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    #self._finish()
                    raise StopIteration

                if not self.pin_memory:
                    item = self._queues[self._next_queue()].get()
                else:
                    item = self.pin_memory_queue.get()

            return item
        except KeyboardInterrupt:
            logging.error("MultiThreadedGenerator: caught exception: {}".format(sys.exc_info()))
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if len(self._threads) == 0:
            logging.debug("starting workers")
            self._queue_loop = 0
            self._end_ctr = 0

            for i in range(self.num_processes):
                self._queues.append(Queue(self.num_cached_per_queue))
                self._threads.append(Process(target=producer, args=(self._queues[i], self.generator, self.transform, i, self.seeds[i])))
                self._threads[-1].daemon = True
                self._threads[-1].start()

            if self.pin_memory:
                self.pin_memory_queue = thrQueue(2)
                self.pin_memory_thread = threading.Thread(target=pin_memory_loop, args=(self._queues, self.pin_memory_queue))
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")

    def _finish(self):
        if len(self._threads) != 0:
            logging.debug("MultiThreadedGenerator: workers terminated")
            for i, thread in enumerate(self._threads):
                thread.terminate()
                self._queues[i].close()
            self._queues = []
            self._threads = []
            self._queue = None
            self._end_ctr = 0
            self._queue_loop = 0

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
