
from multiprocessing import Pool
from default_configs import splitted_4D_out_dir, cropped_output_dir
import numpy as np
import pickle
from preprocessing.cropping import get_patientID_from_cropped_files
from skimage.morphology import label
from collections import OrderedDict
from utils.files_utils import *

class DatasetAnalyzer(object):
    def __init__(self, folder_of_cropped_data, overwrite=True, num_processes=8):
        """
        :param folder_of_cropped_data:
        :param overwrite: 
            default is True. 
            True:recomputed from the data, precomputed values will not be used. 
            False: allow loading of precomputed values.
        """
        self.num_processes = num_processes
        self.overwrite = overwrite
        self.folder_of_cropped_data = folder_of_cropped_data
        self.sizes = self.spacings = None
        self.patient_identifiers = get_patientID_from_cropped_files(self.folder_of_cropped_data)
        assert isfile(join(self.folder_of_cropped_data, "dataset.json")), \
            "dataset.json needs to be in folder_of_cropped_data"
        self.props_per_case_file = join(self.folder_of_cropped_data, "props_per_case.pkl")
        self.intensityproperties_file = join(self.folder_of_cropped_data, "intensityproperties.pkl")

    def load_cropped_properties(self, case_identifier):
        with open(join(self.folder_of_cropped_data, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    @staticmethod
    def _checkall_in_one_region(args):
        seg, regions = args
        res = OrderedDict()
        for r in regions:
            new_seg = np.zeros(seg.shape)
            for c in r:
                new_seg[seg == c] = 1
            labelmap, numlabels = label(new_seg, return_num=True)
            if numlabels != 1:
                res[tuple(r)] = False
            else:
                res[tuple(r)] = True
        return res

    @staticmethod
    def get_class_and_region_sizes(args):
        seg, all_classes, vol_per_voxel = args
        volume_per_class = OrderedDict()
        region_volume_per_class = OrderedDict()
        for c in all_classes:
            region_volume_per_class[c] = []
            volume_per_class[c] = np.sum(seg == c) * vol_per_voxel
            labelmap, numregions = label(seg == c, return_num=True)
            for l in range(1, numregions + 1):
                region_volume_per_class[c].append(np.sum(labelmap == l) * vol_per_voxel)
        return volume_per_class, region_volume_per_class

    def _get_analyzed_seg_classes(self, args):
        """
        1) what class? unique classes;
        2) size distribution of each class;
        3) region size of each class;
        4) check if all in one region;
        :return:
        """
        patient_identifier, all_classes = args
        seg = np.load(join(self.folder_of_cropped_data, patient_identifier) + ".npz")['data'][-1]
        pkl = load_pickle(join(self.folder_of_cropped_data, patient_identifier) + ".pkl")
        vol_per_voxel = np.prod(pkl['itk_spacing'])

        # ad 1)
        unique_classes = np.unique(seg)

        # 4) check if all in one region
        regions = list()
        regions.append(list(all_classes))
        for c in all_classes:
            regions.append((c, ))

        all_in_one_region = self._checkall_in_one_region((seg, regions))

        # 2 & 3) region sizes
        volume_per_class, region_sizes = self.get_class_and_region_sizes((seg, all_classes, vol_per_voxel))

        return unique_classes, all_in_one_region, volume_per_class, region_sizes

    def get_classes(self):
        datasetjson = load_json(join(self.folder_of_cropped_data, "dataset.json"))
        return datasetjson['labels']

    def analyze_segmentations(self):
        class_dct = self.get_classes()
        all_classes = np.array([int(i) for i in class_dct.keys()])
        all_classes = all_classes[all_classes > 0]  # remove background

        if self.overwrite or not isfile(self.props_per_case_file):
            p = Pool(self.num_processes)
            res = p.map(self._get_analyzed_seg_classes, zip(self.patient_identifiers,
                                                            [all_classes] * len(self.patient_identifiers)))
            p.close()
            p.join()

            props_per_patient = OrderedDict()
            for p, (unique_classes, all_in_one_region, voxels_per_class, region_volume_per_class) in \
                    zip(self.patient_identifiers, res):
                props = dict()
                props['has_classes'] = unique_classes
                props['only_one_region'] = all_in_one_region
                props['volume_per_class'] = voxels_per_class
                props['region_volume_per_class'] = region_volume_per_class
                props_per_patient[p] = props

            save_pickle(props_per_patient, self.props_per_case_file)
        else:
            props_per_patient = load_pickle(self.props_per_case_file)
        return class_dct, props_per_patient

    def get_cropped_sizes_and_spacings(self):
        case_identifiers = get_patientID_from_cropped_files(self.folder_of_cropped_data)
        sizes = []
        spacings = []
        for c in case_identifiers:
            properties = self.load_cropped_properties(c)
            sizes.append(properties["size_after_cropping"])
            spacings.append(properties["original_spacing"])

        return sizes, spacings

    def get_modalities(self):
        datasetjson = load_json(join(self.folder_of_cropped_data, "dataset.json"))
        modalities = datasetjson["modality"]
        modalities = {int(k): modalities[k] for k in modalities.keys()}
        return modalities

    def get_cropped_size_reduction(self):
        size_reduction = OrderedDict()
        for p in self.patient_identifiers:
            props = self.load_cropped_properties(p)
            shape_before_crop = props["original_size_of_raw_data"]
            shape_after_crop = props['size_after_cropping']
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            size_reduction[p] = size_red
        return size_reduction

    def _get_voxels(self, args):
        patient_identifier, modality_id = args
        all_data = np.load(join(self.folder_of_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0
        voxels = list(modality[mask][::10]) # no need to take every voxel
        return voxels

    @staticmethod
    def _get_stats(voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def get_intensity(self, num_modalities):
        if self.overwrite or not isfile(self.intensityproperties_file):
            p = Pool(self.num_processes)

            results = OrderedDict()
            for mod_id in range(num_modalities):
                results[mod_id] = OrderedDict()
                v = p.map(self._get_voxels, zip(self.patient_identifiers,
                                                              [mod_id] * len(self.patient_identifiers)))

                w = []
                for iv in v:
                    w += iv

                median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._get_stats(w)

                local_props = p.map(self._get_stats, v)
                props_per_case = OrderedDict()
                for i, pat in enumerate(self.patient_identifiers):
                    props_per_case[pat] = OrderedDict()
                    props_per_case[pat]['median'] = local_props[i][0]
                    props_per_case[pat]['mean'] = local_props[i][1]
                    props_per_case[pat]['sd'] = local_props[i][2]
                    props_per_case[pat]['mn'] = local_props[i][3]
                    props_per_case[pat]['mx'] = local_props[i][4]
                    props_per_case[pat]['percentile_99_5'] = local_props[i][5]
                    props_per_case[pat]['percentile_00_5'] = local_props[i][6]

                results[mod_id]['local_props'] = props_per_case
                results[mod_id]['median'] = median
                results[mod_id]['mean'] = mean
                results[mod_id]['sd'] = sd
                results[mod_id]['mn'] = mn
                results[mod_id]['mx'] = mx
                results[mod_id]['percentile_99_5'] = percentile_99_5
                results[mod_id]['percentile_00_5'] = percentile_00_5

            p.close()
            p.join()
            save_pickle(results, self.intensityproperties_file)
        else:
            results = load_pickle(self.intensityproperties_file)
        return results

    def analyze_dataset(self, collect_intensityproperties=True):
        # get all spacings and sizes
        sizes, spacings = self.get_cropped_sizes_and_spacings()

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        class_dct, segmentation_props_per_patient = self.analyze_segmentations()
        all_classes = np.array([int(i) for i in class_dct.keys()])
        all_classes = all_classes[all_classes > 0]

        # modalities
        modalities = self.get_modalities()

        # collect intensity information
        if collect_intensityproperties:
            intensityproperties = self.get_intensity(len(modalities))
        else:
            intensityproperties = None

        # reduct size by cropping
        size_reductions = self.get_cropped_size_reduction()

        dataset_properties = dict()
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['segmentation_props_per_patient'] = segmentation_props_per_patient
        dataset_properties['class_dct'] = class_dct  # {int: class name}
        dataset_properties['all_classes'] = all_classes
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        save_pickle(dataset_properties, join(self.folder_of_cropped_data, "dataset_properties.pkl"))
        return dataset_properties


if __name__ == "__main__":
    tasks = [i for i in os.listdir(splitted_4D_out_dir) if os.path.isdir(os.path.join(splitted_4D_out_dir, i))]
    tasks.sort()

    t = 'Task_BoneSeg'

    print("\n\n\n", t)
    cropped_out_dir = os.path.join(cropped_output_dir, t)

    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False)
    props = dataset_analyzer.analyze_dataset()
