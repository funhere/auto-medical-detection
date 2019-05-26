#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from builtins import object
import warnings
from warnings import warn
import numpy as np
from abc import ABCMeta, abstractmethod


class DataLoaderBase(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        """
        :param data: a tuple of images (b,c,x,y(,z)) and segmentations (b,c,x,y(,z))
        :param batch_size: 
        :param number_of_threads_in_multithreaded: multiple threads
        """
        warn("should derive from this class to implement your own DataLoader. should overrive self.generate_batch()")
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.gen_train_batch()

    @abstractmethod
    def generate_batch(self):
        '''override this
        Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


if __name__ == '__main__':
    #usage sample:
    class BasicDataLoader(DataLoaderBase):
        """
        data is a tuple of images (b,c,x,y(,z)) and segmentations (b,c,x,y(,z))
        """
    
        def generate_batch(self):
            #Sample randomly from data
            idx = np.random.choice(self._data[0].shape[0], self.batch_size, True, None)
            
            x = np.array(self._data[0][idx])
            y = np.array(self._data[1][idx])
            data_dict = {"data": x,
                         "seg": y}
            return data_dict
