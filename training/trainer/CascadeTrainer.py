from multiprocessing.pool import Pool
import matplotlib
from datasets.data_augmentation.default_data_augmentation import get_default_aug
from training.dataloading.dataset_loading import DataLoader3D, unpack_dataset
from evaluation.evaluator import aggregate_scores
from training.Trainer import Trainer
from models.base_net import DetectionNet
from configs import net_training_out_dir
from utils.exp_utils import store_seg_from_softmax

import numpy as np
from utilities.one_hot_encoding import to_one_hot
import shutil
from utils.files_utils import *

matplotlib.use("agg")


class CascadeTrainer(Trainer):
    def __init__(self, cf, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="Trainer"):
        super(CascadeTrainer, self).__init__(plans_file, fold, output_folder, dataset_directory,
                                                          batch_dice, stage, unpack_data, deterministic)
        if self.output_folder is not None:
            task = self.output_folder.split("/")[-3]
            plans_identifier = self.output_folder.split("/")[-2].split("__")[-1]

            folder_with_segs_prev_stage = join(net_training_out_dir, "3d_lowres",
                                               task, previous_trainer + "__" + plans_identifier, "pred_next_stage")
            if not isdir(folder_with_segs_prev_stage):
                raise RuntimeError(
                    "Cannot run final stage of cascade. Run corresponding 3d_lowres first and predict the "
                    "segmentations for the next stage")
            self.folder_with_segs_from_prev_stage = folder_with_segs_prev_stage
            self.folder_with_segs_from_prev_stage_for_train = join(self.dataset_directory, "segs_prev_stage")
        else:
            self.folder_with_segs_from_prev_stage = None
            self.folder_with_segs_from_prev_stage_for_train = None

    def do_split(self):
        super(CascadeTrainer, self).do_split()
        for k in self.dataset:
            self.dataset[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                               k + "_segFromPrevStage.npz")
            assert isfile(self.dataset[k]['seg_from_prev_stage_file']), \
                "seg from prev stage missing: %s" % (self.dataset[k]['seg_from_prev_stage_file'])
        for k in self.dataset_val:
            self.dataset_val[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                                   k + "_segFromPrevStage.npz")
        for k in self.dataset_tr:
            self.dataset_tr[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                                  k + "_segFromPrevStage.npz")

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 True, oversample_foreground_percent=self.oversample_foreground_percent)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, True,
                                  oversample_foreground_percent=self.oversample_foreground_percent)
        else:
            raise NotImplementedError
        return dl_tr, dl_val

    def process_plans(self, plans):
        super(CascadeTrainer, self).process_plans(plans)
        self.num_input_channels += (self.num_classes - 1)  # for seg from prev stage

    def setup_DA_params(self):
        super(CascadeTrainer, self).setup_DA_params()
        self.data_aug_params['selected_seg_channels'] = [0, 1]
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))
        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        self.data_aug_params['advanced_pyramid_augmentations'] = True

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """
        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()

        self.folder_with_preprocessing_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                  "_stage%d" % self.stage)
        if training:
            # copy segs from prev stage to separate folder and extract them

            if isdir(self.folder_with_segs_from_prev_stage_for_train):
                shutil.rmtree(self.folder_with_segs_from_prev_stage_for_train)

            maybe_mkdir_p(self.folder_with_segs_from_prev_stage_for_train)
            segs_from_prev_stage_files = subfiles(self.folder_with_segs_from_prev_stage, suffix='.npz')
            for s in segs_from_prev_stage_files:
                shutil.copy(s, self.folder_with_segs_from_prev_stage_for_train)

            # if not do this then performance is shit
            if self.unpack_data:
                unpack_dataset(self.folder_with_segs_from_prev_stage_for_train)

            self.folder_with_segs_from_prev_stage = self.folder_with_segs_from_prev_stage_for_train

            self.setup_DA_params()

            if self.folder_with_preprocessing_data is not None:
                self.dl_tr, self.dl_val = self.get_basic_generators()

                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessing_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! ")

                self.tr_gen, self.val_gen = get_default_aug(self.dl_tr, self.dl_val,
                                                                     self.data_aug_params['patch_size_for_spatialtransform'],
                                                                     self.data_aug_params)
                self.write_to_log("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())))
                self.write_to_log("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())))
        else:
            pass
        self.init_net_optimizer_and_scheduler()
        assert isinstance(self.net, DetectionNet)
        self.was_initialized = True

    def validate(self, do_mirroring=True, use_train_mode=False, tiled=True, step=2, save_softmax=True,
                 use_gaussian=True, validation_folder_name='validation'):
        """

        :param do_mirroring:
        :param use_train_mode:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param use_nifti:
        :param save_softmax:
        :param use_gaussian:
        :param use_temporal_models:
        :return:
        """
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)

        if do_mirroring:
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        process_manager = Pool(2)
        results = []

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            data = np.load(self.dataset[k]['data_file'])['data']

            # concat segmentation of previous step
            seg_from_prev_stage = np.load(join(self.folder_with_segs_from_prev_stage,
                                               k + "_segFromPrevStage.npz"))['data'][None]

            transpose_forward = self.plans.get('transpose_forward')
            if transpose_forward is not None:
                data = data.transpose([0] + [i+1 for i in transpose_forward])
                seg_from_prev_stage = seg_from_prev_stage.transpose([0] + [i+1 for i in transpose_forward])

            print(data.shape)
            data[-1][data[-1] == -1] = 0
            data_for_net = np.concatenate((data[:-1], to_one_hot(seg_from_prev_stage[0], range(1, self.num_classes))))
            softmax_pred = self.predict_preprocessing_return_softmax(data_for_net, do_mirroring, 1,
                                                                         use_train_mode, 1, mirror_axes, tiled,
                                                                         True, step, self.patch_size,
                                                                         use_gaussian=use_gaussian)

            if transpose_forward is not None:
                transpose_backward = self.plans.get('transpose_backward')
                softmax_pred = softmax_pred.transpose([0] + [i+1 for i in transpose_backward])

            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

            if save_softmax:
                softmax_fname = join(output_folder, fname + ".npz")
            else:
                softmax_fname = None

            if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.9): # *0.9 just to be save
                np.save(fname + ".npy", softmax_pred)
                softmax_pred = fname + ".npy"
            results.append(process_manager.starmap_async(store_seg_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, 1, None, None, None, softmax_fname, None),
                                                          )
                                                         )
                           )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"), json_name=job_name,
                             json_author="Fabian", json_description="",
                             json_task=task)
