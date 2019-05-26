
from copy import deepcopy
import numpy as np
from analysis.DatasetAnalyzer import DatasetAnalyzer
from utils.analysis_utils import get_pool_and_conv_props_poolLateV2
from analysis.plan_and_preprocess_task import get_lists_of_splitted_dataset, crop
from preprocessing.cropping import get_patientID_from_npz
from preprocessing.preprocessing import GenericPreprocessor
from analysis.configuration import FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, TARGET_SPACING_PERCENTILE, \
    batch_size_covers_max_percent_of_dataset, dataset_min_batch_size_cap, \
    HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0, MIN_SIZE_PER_CLASS_FACTOR, \
    RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD

import shutil
from utils.files_utils import *
from net_architecture.generic_UNet import Generic_UNet
from collections import OrderedDict

class Planner(object):
    def __init__(self, folder_of_cropped_data, preprocessing_out_folder):
        self.folder_of_cropped_data = folder_of_cropped_data
        self.preprocessing_out_folder = preprocessing_out_folder
        self.list_of_cropped_npz_files = subfiles(self.folder_of_cropped_data, True, None, ".npz", True)

        assert isfile(join(self.folder_of_cropped_data, "dataset_properties.pkl")), \
            "folder_of_cropped_data must contain dataset_properties.pkl"
        self.dataset_properties = load_pickle(join(self.folder_of_cropped_data, "dataset_properties.pkl"))

        self.plans_per_stage = OrderedDict()
        self.plans = OrderedDict()
        self.plans_fname = join(self.preprocessing_out_folder, default_plans_identifier + "_plans_3D.pkl")
        self.data_identifier = 'UNet'

    def get_target_spacing(self):
        spacings = self.dataset_properties['all_spacings']

        target = np.percentile(np.vstack(spacings), TARGET_SPACING_PERCENTILE, 0)
        return target

    def save_plans(self):
        with open(self.plans_fname, 'wb') as f:
            pickle.dump(self.plans, f)

    def load_plans(self):
        with open(self.plans_fname, 'rb') as f:
            self.plans = pickle.load(f)

        self.plans_per_stage = self.plans['plans_per_stage']
        self.dataset_properties = self.plans['dataset_properties']

    def do_postprocessing(self):

        print("determining postprocessing...")

        props_per_patient = self.dataset_properties['segmentation_props_per_patient']

        all_region_keys = [i for k in props_per_patient.keys() for i in props_per_patient[k]['only_one_region'].keys()]
        all_region_keys = list(set(all_region_keys))

        store_largest_connected_component = OrderedDict()

        for r in all_region_keys:
            all_results = [props_per_patient[k]['only_one_region'][r] for k in props_per_patient.keys()]
            store_largest_connected_component[tuple(r)] = all(all_results)

        print("Postprocessing: store_largest_connected_component", store_largest_connected_component)

        all_classes = self.dataset_properties['all_classes']
        classes = [i for i in all_classes if i > 0]

        props_per_patient = self.dataset_properties['segmentation_props_per_patient']

        min_size_per_class = OrderedDict()
        for c in classes:
            all_num_voxels = []
            for k in props_per_patient.keys():
                all_num_voxels.append(props_per_patient[k]['volume_per_class'][c])
            if len(all_num_voxels) > 0:
                min_size_per_class[c] = np.percentile(all_num_voxels, 1) * MIN_SIZE_PER_CLASS_FACTOR
            else:
                min_size_per_class[c] = np.inf

        min_region_size_per_class = OrderedDict()
        for c in classes:
            region_sizes = [l for k in props_per_patient for l in props_per_patient[k]['region_volume_per_class'][c]]
            if len(region_sizes) > 0:
                min_region_size_per_class[c] = min(region_sizes)
                # we don't need that line but better safe than sorry, right?
                min_region_size_per_class[c] = min(min_region_size_per_class[c], min_size_per_class[c])
            else:
                min_region_size_per_class[c] = 0

        print("Postprocessing: min_size_per_class", min_size_per_class)
        print("Postprocessing: min_region_size_per_class", min_region_size_per_class)
        return store_largest_connected_component, min_size_per_class, min_region_size_per_class

    def plan_exps(self):
        architecture_input_voxels = np.prod(Generic_UNet.DEFAULT_PATCH_SIZE_3D)

        def get_stage_properties(current_spacing, original_spacing, original_shape, num_cases,
                                     num_modalities, num_classes):
            """
            Computation of input patch size starts out with the new median shape (in voxels) of a dataset. This is
            opposed to prior experiments where I based it on the median size in mm. The rationale behind this is that
            for some organ of interest the acquisition method will most likely be chosen such that the field of view and
            voxel resolution go hand in hand to show the doctor what they need to see. This assumption may be violated
            for some modalities with anisotropy (cine MRI) but we will have t live with that. In future experiments I
            will try to 1) base input patch size match aspect ratio of input size in mm (instead of voxels) and 2) to
            try to enforce that we see the same 'distance' in all directions (try to maintain equal size in mm of patch)
            :param current_spacing:
            :param original_spacing:
            :param original_shape:
            :param num_cases:
            :return:
            """
            new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
            dataset_num_voxels = np.prod(new_median_shape) * num_cases

            # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
            # input_patch_size = new_median_shape

            # compute how many voxels are one mm
            input_patch_size = 1 / np.array(current_spacing)

            # normalize voxels per mm
            input_patch_size /= input_patch_size.mean()

            # create an isotropic patch of size 512x512x512mm
            input_patch_size *= 1 / min(input_patch_size) * 512 # to get a starting value
            input_patch_size = np.round(input_patch_size).astype(int)

            # clip it to the median shape of the dataset because patches larger then that make not much sense
            input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]

            net_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp,  \
                shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(input_patch_size,
                                                                     FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                     Generic_UNet.MAX_NUMPOOL_3D, current_spacing)

            ref = Generic_UNet.use_this_for_batch_size_computation_3D
            here = Generic_UNet.compute_vram_consumption(new_shp, net_num_pool_per_axis,
                                                                Generic_UNet.BASE_NUM_FEATURES_3D,
                                                                Generic_UNet.MAX_NUM_FILTERS_3D, num_modalities,
                                                                num_classes,
                                                                pool_op_kernel_sizes)
            while here > ref:
                argsrt = np.argsort(new_shp / new_median_shape)[::-1]
                pool_fct_per_axis = np.prod(pool_op_kernel_sizes, 0)
                bottleneck_size_per_axis = new_shp / pool_fct_per_axis
                shape_must_be_divisible_by = [shape_must_be_divisible_by[i]
                                              if bottleneck_size_per_axis[i] > 4
                                              else shape_must_be_divisible_by[i] / 2
                                              for i in range(len(bottleneck_size_per_axis))]
                new_shp[argsrt[0]] -= shape_must_be_divisible_by[argsrt[0]]

                # we have to recompute numpool now:
                net_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
                shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(new_shp,
                                                                     FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                     Generic_UNet.MAX_NUMPOOL_3D, current_spacing)

                here = Generic_UNet.compute_vram_consumption(new_shp, net_num_pool_per_axis,
                                                                    Generic_UNet.BASE_NUM_FEATURES_3D,
                                                                    Generic_UNet.MAX_NUM_FILTERS_3D, num_modalities,
                                                                    num_classes, pool_op_kernel_sizes)
                print(new_shp)

            input_patch_size = new_shp

            batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what wirks with 128**3
            batch_size = int(np.floor(max(ref / here, 1) * batch_size))

            # check if batch size is too large
            max_batch_size = np.round(batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                      np.prod(input_patch_size)).astype(int)
            max_batch_size = max(max_batch_size, dataset_min_batch_size_cap)
            batch_size = min(batch_size, max_batch_size)

            do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[0]) > RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD

            plan = {
                'batch_size': batch_size,
                'num_pool_per_axis': net_num_pool_per_axis,
                'patch_size': input_patch_size,
                'median_patient_size_in_voxels': new_median_shape,
                'current_spacing': current_spacing,
                'original_spacing': original_spacing,
                'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
                'pool_op_kernel_sizes': pool_op_kernel_sizes,
                'conv_kernel_sizes': conv_kernel_sizes,
            }
            return plan

        use_nonzero_mask_for_normalization = self.use_norm_mask()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = list()

        self.plans_per_stage.append(get_stage_properties(target_spacing, target_spacing, median_shape,
                                                             len(self.list_of_cropped_npz_files),
                                                             num_modalities, len(all_classes) + 1))
        if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels']) / \
                architecture_input_voxels < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:
            more = False
        else:
            more = True

        if more:
            # if we are doing more than one stage then we want the lowest stage to have exactly
            # HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 (this is 4 by default so the number of voxels in the
            # median shape of the lowest stage must be 4 times as much as the net can process at once (128x128x128 by
            # default). Problem is that we are downsampling higher resolution axes before we start downsampling the
            # out-of-plane axis. We could probably/maybe do this analytically but I am lazy, so here
            # we do it the dumb way
            lowres_stage_spacing = deepcopy(target_spacing)
            num_voxels = np.prod(median_shape)
            while num_voxels > HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 * architecture_input_voxels:
                max_spacing = max(lowres_stage_spacing)
                if np.any((max_spacing / lowres_stage_spacing) > 2):
                    lowres_stage_spacing[(max_spacing / lowres_stage_spacing) > 2] \
                        *= 1.01
                else:
                    lowres_stage_spacing *= 1.01
                num_voxels = np.prod(target_spacing / lowres_stage_spacing * median_shape)

            new = get_stage_properties(lowres_stage_spacing, target_spacing, median_shape,
                                                                 len(self.list_of_cropped_npz_files),
                                           num_modalities, len(all_classes) + 1)

            if 1.5 * np.prod(new['median_patient_size_in_voxels']) < np.prod(self.plans_per_stage[0]['median_patient_size_in_voxels']):
                self.plans_per_stage.append(new)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        print(self.plans_per_stage)

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
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage}

        self.plans = plans
        self.save_plans()

    def do_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "CT"
            else:
                schemes[i] = "nonCT"
        return schemes

    def save_cropped_properties(self, case_identifier, properties):
        with open(join(self.folder_of_cropped_data, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def load_cropped_properties(self, case_identifier):
        with open(join(self.folder_of_cropped_data, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def use_norm_mask(self):
        # only use the nonzero mask for normalization of the cropping based on it resulted in a decrease in
        # image size (this is an indication that the data is something like brats/isles and then we want to
        # normalize in the brain region only)
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))
        use_nonzero_mask_for_norm = OrderedDict()
        
        for i in range(num_modalities):
            if "CT" in modalities[i]:
                use_nonzero_mask_for_norm[i] = False
            else:
                all_size_reductions = []
                for k in self.dataset_properties['size_reductions'].keys():
                    all_size_reductions.append(self.dataset_properties['size_reductions'][k])
    
                if np.median(all_size_reductions) < 3/4.:
                    print("using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = True
                else:
                    print("not using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = False

        # write use_nonzero_mask_for_norm into properties -> will be needed for data augmentation TODO
        for c in self.list_of_cropped_npz_files:
            case_identifier = get_patientID_from_npz(c)
            properties = self.load_cropped_properties(case_identifier)
            properties['use_nonzero_mask_for_norm'] = use_nonzero_mask_for_norm
            self.save_cropped_properties(case_identifier, properties)
        use_nonzero_mask_for_normalization = use_nonzero_mask_for_norm
        return use_nonzero_mask_for_normalization

    def add_normalization_to_patients(self):
        """
        Used for test set preprocessing
        """
        for c in self.list_of_cropped_npz_files:
            case_identifier = get_patientID_from_npz(c)
            properties = self.load_cropped_properties(case_identifier)
            properties['use_nonzero_mask_for_norm'] = self.plans['use_mask_for_norm']
            self.save_cropped_properties(case_identifier, properties)

    def do_preprocessing(self, num_threads):
        if os.path.isdir(join(self.preprocessing_out_folder, "gt_segmentations")):
            shutil.rmtree(join(self.preprocessing_out_folder, "gt_segmentations"))
        shutil.copytree(join(self.folder_of_cropped_data, "gt_segmentations"), 
                        join(self.preprocessing_out_folder, "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        preprocessor = GenericPreprocessor(normalization_schemes, use_nonzero_mask_for_normalization,
                                           intensityproperties)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (8, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        preprocessor.run(target_spacings, self.folder_of_cropped_data, self.preprocessing_out_folder,
                         self.plans['data_identifier'], num_threads)


if __name__ == "__main__":
    t = "Task_BrainTumorIntern"

    cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)

    threads = 64

    crop(t, False, threads)

    print("\n\n\n", t)
    cropped_out_dir = os.path.join(cropped_output_dir, t)
    preprocessing_out_dir = os.path.join(preprocessing_output_dir, t)
    splitted_4D_out_dir_task = os.path.join(splitted_4D_out_dir, t)
    lists, modalities = get_lists_of_splitted_dataset(splitted_4D_out_dir_task)

    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=threads)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties=True) # this will write output files that will be used by the Planner

    maybe_mkdir_p(preprocessing_out_dir)
    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_out_dir)
    shutil.copy(join(splitted_4D_out_dir, t, "dataset.json"), preprocessing_out_dir)

    print("number of threads: ", threads, "\n")

    exp_planner = Planner(cropped_out_dir, preprocessing_out_dir)
    exp_planner.plan_exps()
    exp_planner.do_preprocessing(threads)
