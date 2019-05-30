
import argparse
from utils.files_utils import *
from utils.exp_utils import prep_exp
from run.default_configuration import get_default_configuration
from default_configs import default_plans_identifier
from training.predict_next import predict_next
from training.trainer.Trainer import Trainer
from training.trainer.CascadeTrainer import CascadeTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net")
    parser.add_argument("net_trainer")
    parser.add_argument("task")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier", default=default_plans_identifier, required=False)
    parser.add_argument("-u", "--unpack_data", help="Leave it as 1, development only", required=False, default=1,
                        type=int)
    parser.add_argument("--ndet", 
                        help="nondeterministic training, it allows cudnn.benchmark which will can increase performance."
                        "default training is deterministic.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", 
                        help="if set then UNet will export npz files of predicted segmentations in the vlaidation as well.")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true", help="not used, for analysis only.")
    parser.add_argument("--valbest", required=False, default=False, action="store_true", help="hands off. for analysis only.")
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--exp_source', type=str, default='experiments/demo_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')
    
    args = parser.parse_args()

    task = args.task
    fold = args.fold
    net = args.net
    net_trainer = args.net_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    unpack = args.unpack_data
    deterministic = not args.ndet
    valbest = args.valbest

    if unpack == 0:
        unpack = False
    elif unpack == 1:
        unpack = True
    else:
        raise ValueError("Unexpected value for -u/--unpack_data: %s. Use 1 or 0." % str(unpack))

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    cf = prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=True, use_stored_settings=True)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(net, task, net_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in training.trainer")

    if net == "3d_cascade_fullres":
        assert issubclass(trainer_class, CascadeTrainer), "If running 3d_cascade_fullres then your " \
                           "trainer class must be derived from CascadeTrainer."
    else:
        assert issubclass(trainer_class, Trainer), "net_trainer was found but is not derived from Trainer"

    trainer = trainer_class(cf, plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=unpack, deterministic=deterministic)

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            trainer.do_training()
        elif not valbest:
            trainer.load_latest_checkpoint(train=False)

        if valbest:
            trainer.load_best_checkpoint(train=False)
            val_folder = "validation_best_epoch"
        else:
            val_folder = "validation"

        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)

        if net == '3d_lowres':
            trainer.load_best_checkpoint(False)
            print("predicting segmentations for the next stage of the cascade")
            predict_next(cf, trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
