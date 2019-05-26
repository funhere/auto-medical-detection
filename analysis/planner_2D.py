
import shutil, os
from analysis.planner_3D import Planner
from bins.analyze_and_preprocess import get_lists_of_splitted_dataset
from preprocessing.preprocessor import Preprocessor2D
from config.default_configs import *
from utils.files_utils import *

from net_architecture.generic_UNet import Generic_UNet
import numpy as np
from utils.analysis_utils import get_pool_and_conv_props


class Planner2D(Planner):
    def __init__(self, folder_of_cropped_data, preprocessing_out_folder):
        super(Planner2D, self).__init__(folder_of_cropped_data,
                                                  preprocessing_out_folder)
        self.data_identifier = "UNet_2D"
        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]
        self.plans_fname = join(self.preprocessing_out_folder, default_plans_identifier + "_plans_2D.pkl")


    def load_plans(self):
        self.plans = load_pickle(self.plans_fname)

        self.plans_per_stage = self.plans['plans_per_stage']
        self.dataset_properties = self.plans['dataset_properties']
        self.transpose_forward = self.plans['transpose_forward']
        self.transpose_backward = self.plans['transpose_backward']

    def plan_exps(self):
        def get_stage_properties(current_spacing, original_spacing, original_shape, num_cases, transpose_forward,
                                     num_modalities, num_classes):
            current_spacing_transposed = np.array([current_spacing[i] for i in transpose_forward])[1:]

            new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
            dataset_num_voxels = np.prod(new_median_shape) * num_cases
            input_patch_size = new_median_shape[transpose_forward][1:]

            net_numpool, net_pool_kernel_sizes, net_conv_kernel_sizes, input_patch_size, \
                shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing_transposed, input_patch_size,
                                                                     FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                     Generic_UNet.MAX_NUMPOOL_2D)

            estimated_gpu_ram_consumption = Generic_UNet.compute_vram_consumption(input_patch_size,
                                                                                         net_numpool,
                                                                                         Generic_UNet.BASE_NUM_FEATURES_2D,
                                                                                         Generic_UNet.MAX_FILTERS_2D,
                                                                                         num_modalities, num_classes,
                                                                                         net_pool_kernel_sizes)

            batch_size = int(np.floor(Generic_UNet.use_this_for_batch_size_computation_2D /
                                      estimated_gpu_ram_consumption * Generic_UNet.DEFAULT_BATCH_SIZE_2D))
            if batch_size < dataset_min_batch_size_cap:
                raise RuntimeError("Unsupported patches size. patch-based solution will be implemented later.")

            # check if batch size is too large (more than 5 % of dataset)
            max_batch_size = np.round(batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                      np.prod(input_patch_size)).astype(int)
            batch_size = min(batch_size, max_batch_size)

            plan = {
                'batch_size': batch_size,
                'num_pool_per_axis': net_numpool,
                'patch_size': input_patch_size,
                'median_patient_size_in_voxels': new_median_shape,
                'current_spacing': current_spacing,
                'original_spacing': original_spacing,
                'pool_op_kernel_sizes': net_pool_kernel_sizes,
                'conv_kernel_sizes': net_conv_kernel_sizes,
                'do_dummy_2D_data_aug': False
            }
            return plan

        use_nonzero_mask_for_normalization = self.use_norm_mask()
        print("Are you using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)

        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = np.array([np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)])

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]
        new_shapes = new_shapes[:, self.transpose_forward]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("Don't want feature maps smaller than ", FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = []

        self.plans_per_stage.append(get_stage_properties(target_spacing, target_spacing, median_shape,
                                                              num_cases=len(self.list_of_cropped_npz_files),
                                                              transpose_forward=self.transpose_forward,
                                                             num_modalities=num_modalities,
                                                             num_classes=len(all_classes) + 1))

        print(self.plans_per_stage)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        normalization_schemes = self.do_normalization_scheme()
        store_largest_connected_component, min_size_per_class, min_region_size_per_class = \
            self.do_postprocessing()

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessing_data_folder': self.preprocessing_out_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': Generic_UNet.BASE_NUM_FEATURES_3D,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': store_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage}

        self.plans = plans
        self.save_plans()

    def do_preprocessing(self, num_threads):
        if os.path.isdir(join(self.preprocessing_out_folder, "gt_segmentations")):
            shutil.rmtree(join(self.preprocessing_out_folder, "gt_segmentations"))
        shutil.copytree(join(self.folder_of_cropped_data, "gt_segmentations"), join(self.preprocessing_out_folder,
                                                                                      "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        preprocessor = Preprocessor2D(normalization_schemes, use_nonzero_mask_for_normalization,
                                           intensityproperties, self.transpose_forward[0])
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        preprocessor.run(target_spacings, self.folder_of_cropped_data, self.preprocessing_out_folder,
                         self.plans['data_identifier'], num_threads)

if __name__ == "__main__":
    t = "Task_BoneSeg"

    print("\n\n\n", t)
    cropped_out_dir = os.path.join(cropped_output_dir, t)
    preprocessing_out_dir = os.path.join(preprocessing_output_dir, t)
    splitted_4D_out_dir_task = os.path.join(splitted_4D_out_dir, t)
    lists, modalities = get_lists_of_splitted_dataset(splitted_4D_out_dir_task)

    # need to be careful with RAM usage
    if t in ["Task_LITS", "Task_Liver", "Task_BoneSegOrigs", "Task_BoneSeg"]:
        threads = 3
    elif t in ["Task_LungIntern", "Task_FibroticLungSeg", "Task_Lung", "Task_HepaticVessel"]:
        threads = 6
    else:
        threads = 8

    print("number of threads: ", threads, "\n")

    print("\n\n\n", t)
    exp_planner = Planner2D(cropped_out_dir, preprocessing_out_dir, threads)
    exp_planner.plan_exps()
    exp_planner.do_preprocessing()
