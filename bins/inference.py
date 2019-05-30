
import argparse
from inference.predictor import predict_group
from default_configs import default_plans_identifier, net_training_out_dir
from utils.files_utils import *
from utils.exp_utils import prep_exp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', 
                        help="Should contain all modalities for each patient in the correct order(same as training).", 
                             required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', 
                        help='task name, required.',
                        default=default_plans_identifier, required=True)

    parser.add_argument('-tr', '--unet_trainer', help='UNet trainer class. Default: Trainer', required=False,
                        default='Trainer')
    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)
    parser.add_argument('-p', '--plans_identifier', help='plans ID',
                        default=default_plans_identifier, required=False)

    parser.add_argument('-f', '--folds', nargs='+', default='None', 
                        help="folds to use for prediction. Default is None ")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true', 
                        help="use this if you want to ensemble")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None', 
                        help="if model is the highres, need to use -l to specify where the segmentations of the "
                         "corresponding lowres unet are. and required to do a prediction")
    parser.add_argument("--part_id", type=int, required=False, default=0, 
                        help="Used to parallelize the prediction of the folder over several GPUs.")
    parser.add_argument("--num_parts", type=int, required=False, default=1, 
                        help="Used to parallelize the prediction of the folder over several GPUs.")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, 
                        help="Determines many background processes will be used for data preprocessing. Default: 6")
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, 
                        help="Determines many background processes will be used for segmentation export. Default: 2")
    parser.add_argument("--tta", required=False, type=int, default=1, 
                        help="test time data augmentation. 0: disable; (e.g. speedup of factor 4(2D)/8(3D)).")
    parser.add_argument("--overwrite_existing", required=False, type=int, default=1, 
                        help="Set this to 0 if you need to resume a previous prediction. ")
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')


    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    tta = args.tta
    overwrite = args.overwrite_existing
    cf = prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=True, use_stored_settings=True)
    
    output_folder_name = join(net_training_out_dir, args.model, args.task_name, args.unet_trainer + "__" +
                              args.plans_identifier)
    print("using model stored in ", output_folder_name)
    assert isdir(output_folder_name), "model output folder not found: %s" % output_folder_name

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unexpected value for tta, Use 1 or 0")

    if overwrite == 0:
        overwrite = False
    elif overwrite == 1:
        overwrite = True
    else:
        raise ValueError("Unexpected value for overwrite, Use 1 or 0")

    predict_group(cf, output_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, tta,
                        overwrite_existing=overwrite)
    

if __name__ == "__main__":
    main()
