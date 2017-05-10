import argparse
import csv
from functools import wraps
from itertools import cycle, islice
from math import ceil
import datetime
import json
import os
import shutil
import types
import numpy as np
import tensorflow as tf
import keras.backend as K
from collections import Iterable, OrderedDict
from keras.optimizers import Adam
from keras.callbacks import (Callback, TensorBoard, EarlyStopping,
                             ModelCheckpoint, CSVLogger, ProgbarLogger)

if 'get_ipython' in vars() and get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from keras_tqdm import TQDMNotebookCallback as Progbar
else:
    from keras_tqdm import TQDMCallback
    import sys
    class Progbar(TQDMCallback):  # redirect TQDMCallback to stdout
        def __init__(self):
            TQDMCallback.__init__(self)
            self.output_file = sys.stdout


class LogDirLogger(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir 

    def on_epoch_begin(self, epoch, logs=None):
        print('\n' + self.log_dir + '\n')


class TimedCSVLogger(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', 'time'] + self.keys,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch, 'time': str(datetime.datetime.now())})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.diff(T, axis=1), np.zeros(T.shape[0])]


def lags_to_times(dT):
    """(N x n_step) matrix of lags -> (N x n_step) matrix of times
    First time is assumed to be zero.
    """
    assert dT.ndim == 2, "dT must be an (N x n_step) matrix"
    return np.c_[np.zeros(dT.shape[0]), np.cumsum(dT[:,:-1], axis=1)]


def noisify_samples(inputs, outputs, errors, batch_size=500, sample_weight=None):
    """
    inputs: {'main_input': X, 'aux_input': X[:, :, [1]]}
    outputs: X[:, :, [1]]
    errors: X_raw[:, :, 2]
    """
    if sample_weight is None:
        sample_weight = np.ones(errors.shape)
    X = inputs['main_input']
    X_aux = inputs['aux_input']
    shuffle_inds = np.arange(len(X))
    while True:
        # New epoch
        np.random.shuffle(shuffle_inds)
        noise = errors * np.random.normal(size=errors.shape)
        X_noisy = X.copy()
        X_noisy[:, :, 1] += noise
        # Re-scale to have mean 0 and std dev 1; TODO make this optional
        X_noisy[:, :, 1] -= np.atleast_2d(np.nanmean(X_noisy[:, :, 1], axis=1)).T
        X_noisy[:, :, 1] /= np.atleast_2d(np.std(X[:, :, 1], axis=1)).T

        for i in range(ceil(len(X) / batch_size)):
            inds = shuffle_inds[(i * batch_size):((i + 1) * batch_size)]
            yield ([X_noisy[inds], X_aux[inds]], X_noisy[inds, :, 1:2], sample_weight[inds])


def parse_model_args(arg_dict=None):
    """Parse command line arguments and optionally combine with values in `arg_dict`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--loss_weights", type=float, nargs='*')
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--decode_type", type=str, default=None)
    parser.add_argument("--decode_layers", type=int, default=None)
    parser.add_argument("--gpu_frac", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str)
    parser.add_argument("--data_type", type=str, default='sinusoid')
    parser.add_argument("--N_train", type=int, default=50000)
    parser.add_argument("--N_test", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=200)
    parser.add_argument("--n_max", type=int, default=200)
    parser.add_argument('--even', dest='even', action='store_true')
    parser.add_argument('--uneven', dest='even', action='store_false')
    parser.add_argument('--no_train', dest='no_train', action='store_true')
    parser.add_argument('--embedding', type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--pool', type=int, default=None)
    parser.add_argument("--first_N", type=int, default=None)
    parser.add_argument("--m_max", type=float, default=20.)
    parser.add_argument("--lomb_score", type=float, default=None)
    parser.add_argument("--ss_resid", type=float, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--finetune_rate', type=float, default=None)
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    parser.add_argument("--survey_files", type=str, nargs='*')
    parser.add_argument('--noisify', dest='noisify', action='store_true')
    parser.add_argument('--period_fold', dest='period_fold', action='store_true')
    parser.set_defaults(even=False, bidirectional=False, noisify=False,
                        period_fold=False)
    # Don't read argv if arg_dict present
    args = parser.parse_args(None if arg_dict is None else [])

    if arg_dict:  # merge additional arguments w/ defaults
        args = argparse.Namespace(**{**args.__dict__, **arg_dict})

    required_args = ['size', 'num_layers', 'drop_frac', 'lr', 'model_type', 'sim_type',
                     'n_min', 'n_max']
    for key in required_args:
        if getattr(args, key) is None:
            parser.error("Missing argument {}".format(key))

    return args


def get_run_id(model_type, size, num_layers, lr, drop_frac=0.0, embedding=None,
               decode_type=None, decode_layers=None, bidirectional=False, **kwargs):
    """Generate unique ID from model params."""
    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(model_type, size, num_layers, lr,
                                                int(100 * drop_frac)).replace('e-', 'm')
    if embedding:
        run += '_emb{}'.format(embedding)
    if decode_type:
        run += '_decode{}'.format(decode_type)
        if decode_layers:
            run += '_x{}'.format(decode_layers)
    if bidirectional:
        run += '_bidir'

    return run


def limited_memory_session(gpu_frac):
    if gpu_frac <= 0.0:
        K.set_session(tf.Session())
    else:
        gpu_opts = tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac))
        K.set_session(tf.Session(config=gpu_opts))


def train_and_log(X, Y, run, model, nb_epoch, batch_size, lr, loss, sim_type, metrics=[],
                  sample_weight=None, no_train=False, patience=20, finetune_rate=None,
                  validation_split=0.2, validation_data=None, gpu_frac=None,
                  noisify=False, errors=None, pretrain_weights=None, **kwargs):
    """Train model and write logs/weights to `keras_logs/{run_id}/`.
    
    If weights already existed, they will be loaded and training will be skipped.
    """
    optimizer = Adam(lr=lr if not finetune_rate else finetune_rate)
    if gpu_frac is not None:
        limited_memory_session(gpu_frac)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  sample_weight_mode='temporal' if sample_weight is not None else None)

    log_dir = os.path.join(os.getcwd(), 'keras_logs', sim_type, run)
    weights_path = os.path.join(log_dir, 'weights.h5')
    loaded = False
    if os.path.exists(weights_path):
        print("Loading {}...".format(weights_path))
        history = []
        model.load_weights(weights_path)
        loaded = True
    elif no_train or finetune_rate:
        raise FileNotFoundError("No weights found in {}.".format(log_dir))

    if finetune_rate:  # write logs to new directory
        log_dir += "_ft{:1.0e}".format(finetune_rate).replace('e-', 'm')

    if (not loaded or finetune_rate) and not no_train:
        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir)
        param_log = {key: value for key, value in locals().items()}
        param_log.update(kwargs)
        param_log = {k: v for k, v in param_log.items()
                     if k not in ['X', 'Y', 'model', 'optimizer', 'sample_weight',
                                  'kwargs', 'validation_data', 'errors']
                        and not isinstance(v, types.FunctionType)}
        json.dump(param_log, open(os.path.join(log_dir, 'param_log.json'), 'w'),
                  sort_keys=True, indent=2)
        if pretrain_weights:
            model.load_weights(pretrain_weights, by_name=True)
        if not noisify:
            history = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size,
                                sample_weight=sample_weight,
                                callbacks=[Progbar(),
                                           TensorBoard(log_dir=log_dir, write_graph=False),
                                           TimedCSVLogger(os.path.join(log_dir, 'training.csv'), append=True),
#                                           EarlyStopping(patience=patience),
                                           ModelCheckpoint(weights_path, save_weights_only=True),
                                           LogDirLogger(log_dir)], verbose=False,
                                validation_split=validation_split,
                                validation_data=validation_data)
        else:
            history = model.fit_generator(noisify_samples(X, Y, errors, batch_size,
                                                          sample_weight),
                                          samples_per_epoch=len(Y), nb_epoch=nb_epoch,
                                          callbacks=[Progbar(),
                                                     TensorBoard(log_dir=log_dir,
                                                                 write_graph=False),
                                                     TimedCSVLogger(os.path.join(log_dir,
                                                                                 'training.csv'),
                                                                    append=True),
                                                     ModelCheckpoint(weights_path,
                                                                     save_weights_only=True),
                                                     LogDirLogger(log_dir)],
                                          verbose=True,
                                          validation_data=validation_data)
    return history
