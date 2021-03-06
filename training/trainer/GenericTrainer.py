from _warnings import warn
import matplotlib
from utils.files_utils import *
from sklearn.model_selection import KFold
matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from datetime import datetime
import torch.backends.cudnn as cudnn
from abc import abstractmethod


class GenericTrainer(object):
    def __init__(self, cf, deterministic=True):
        """
        A generic class that can train almost any neural net (RNNs excluded). 

        What you need to override:
        - __init__
        - initialize
        - do_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        if deterministic:
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.cf = cf
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        self.fold = None
        self.loss = None
        self.dataset_directory = None

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = None  # these can be None for inference mode
        self.dataset_tr = self.dataset_val = None 
        
        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new

        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.save_every = 50
        self.save_latest_only = True
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 250
        self.num_val_batches_per_epoch = 50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = [] # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic


    @abstractmethod
    def initialize(self, training=True):


    @abstractmethod
    def load_dataset(self):
        pass

    def do_split(self):
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.write_to_log("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]


    def plot_progress(self):

        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(self.all_val_losses):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            print("failed to plot: ", sys.exc_info())

    def write_to_log(self, *args, also_print_to_console=True):
        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                         (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("failed to log: ", sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def store_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics)},
            fname)
        self.write_to_log("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train=train)
        all_checkpoints = [i for i in os.listdir(self.output_folder) if i.endswith(".model") and i.find("_ep_") != -1]
        if len(all_checkpoints) == 0:
            return self.load_best_checkpoint(train=train)
        corresponding_epochs = [int(i.split("_")[-1].split(".")[0]) for i in all_checkpoints]
        checkpoint = all_checkpoints[np.argmax(corresponding_epochs)]
        self.load_checkpoint(join(self.output_folder, checkpoint), train=train)

    def load_checkpoint(self, fname, train=True):
        print("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, saved_model, train=True):
        """
        used for if the checkpoint is already in ram
        :param saved_model:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.net.state_dict().keys())
        # if state dict comes form nn.DataParallel but use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                key = key[7:]
            new_state_dict[key] = value
        self.net.load_state_dict(new_state_dict)
        self.epoch = saved_model['epoch']
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = saved_model['plot_stuff']

    def do_training(self):
        if cudnn.benchmark and cudnn.deterministic:
            warn("Both torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark are set True,"
                 "this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        maybe_mkdir_p(self.output_folder)

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.write_to_log("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.net.train()
            for b in range(self.num_batches_per_epoch):
                l = self.run_iteration(self.tr_gen, True)
                l_cpu = l.data.cpu().numpy()
                train_losses_epoch.append(l_cpu)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.write_to_log("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.net.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l.data.cpu().numpy())
                self.all_val_losses.append(np.mean(val_losses))
                self.write_to_log("val loss (train=False): %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.net.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l.data.cpu().numpy())
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.write_to_log("val loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            epoch_end_time = time()
            self.write_to_log("This epoch took %f s" % (epoch_end_time-epoch_start_time))

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()
            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
        self.store_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def check_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.write_to_log("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def check_store_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.epoch % self.save_every == (self.save_every - 1):
            self.write_to_log("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.store_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.store_checkpoint(join(self.output_folder, "model_latest.model"))
            self.write_to_log("done")

    def update_eval_criterion_MA(self):

        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                            1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                            1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            self.write_to_log("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            self.write_to_log("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.write_to_log("saving best epoch checkpoint...")
                self.store_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                self.write_to_log("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                self.write_to_log("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.write_to_log("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    self.write_to_log("My patience ended")
                    continue_training = False
            else:
                self.write_to_log(
                    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def on_epoch_end(self):
        # update self.all_val_eval_metrics
        self.finish_online_evaluation() 

        self.plot_progress()

        self.check_update_lr()

        self.check_store_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def run_iteration(self, data_generator, do_backprop=True, do_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        self.optimizer.zero_grad()

        output = self.net(data)
        l = self.loss(output, target)

        if do_online_evaluation:
            self.do_online_evaluation(output, target)

        if do_backprop:
            l.backward()
            self.optimizer.step()

        return l

    def do_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        refer and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param num_iters:
        :param init_value:
        :param final_value:
        :param beta:
        :return:
        """
        import math
        mult = (final_value / init_value) ** (1/num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, do_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss
            smoothed_loss = avg_loss / (1 - beta**batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        import matplotlib.pyplot as plt
        lrs = [10 ** i for i in log_lrs]
        fig = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses
