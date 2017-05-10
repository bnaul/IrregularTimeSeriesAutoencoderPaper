import glob
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, MaxPooling1D, SimpleRNN)
import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder
from survey_autoencoder import preprocess, main as survey_autoencoder
from light_curve import LightCurve


def main(args=None):
    args = ku.parse_model_args(args)

    args.loss = 'categorical_crossentropy'

    np.random.seed(0)

    if not args.survey_files:
        raise ValueError("No survey files given")
    classes = ['RR_Lyrae_FM', 'W_Ursae_Maj', 'Classical_Cepheid',
               'Beta_Persei', 'Semireg_PV']
    lc_lists = [joblib.load(f) for f in args.survey_files]
    combined = [lc for lc_list in lc_lists for lc in lc_list]
    combined = [lc for lc in combined if lc.label in classes]
    if args.lomb_score:
        combined = [lc for lc in combined if lc.best_score >= args.lomb_score]
    split = [el for lc in combined for el in lc.split(args.n_min, args.n_max)]
    if args.period_fold:
        for lc in split:
            lc.period_fold()
    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]

    classnames, y_inds = np.unique([lc.label for lc in split], return_inverse=True)
    Y = to_categorical(y_inds, len(classnames))

    X_raw = pad_sequences(X_list, value=np.nan, dtype='float', padding='post')
    X, means, scales, wrong_units = preprocess(X_raw, args.m_max)
    Y = Y[~wrong_units]

    # Remove errors
    X = X[:, :, :2]

    if args.even:
        X = X[:, :, 1:]

#    shuffled_inds = np.random.permutation(np.arange(len(X)))
#    train = np.sort(shuffled_inds[:args.N_train])
#    valid = np.sort(shuffled_inds[args.N_train:])
    train, valid = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_list, y_inds))[0]

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN,
                       'conv': Conv1D}#, 'atrous': AtrousConv1D, 'phased': PhasedLSTM}

#    if args.pretrain:
#        auto_args = {k: v for k, v in args.__dict__.items() if k != 'pretrain'}
#        auto_args['sim_type'] = args.pretrain
##        auto_args['no_train'] = True
#        auto_args['epochs'] = 1; auto_args['loss'] = 'mse'; auto_args['batch_size'] = 32; auto_args['sim_type'] = 'test'
#        _, _, auto_model, _ = survey_autoencoder(auto_args)
#        for layer in auto_model.layers:
#            layer.trainable = False
#        model_input = auto_model.input[0]
#        encode = auto_model.get_layer('encoding').output
#    else:
#        model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
#        encode = encoder(model_input, layer=model_type_dict[args.model_type],
#                         output_size=args.embedding, **vars(args))
    model_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    encode = encoder(model_input, layer=model_type_dict[args.model_type],
                     output_size=args.embedding, **vars(args))

    scale_param_input = Input(shape=(2,), name='scale_params')
    merged = merge([encode, scale_param_input], mode='concat')

    out = Dense(args.size + 2, activation='relu')(merged)
    out = Dense(Y.shape[-1], activation='softmax')(out)
    model = Model([model_input, scale_param_input], out)

    run = ku.get_run_id(**vars(args))
    if args.pretrain:
        for layer in model.layers:
            layer.trainable = False
        pretrain_weights = os.path.join('keras_logs', args.pretrain, run, 'weights.h5')
    else:
        pretrain_weights = None

    history = ku.train_and_log([X[train], np.c_[means, scales][train]], Y[train],
                               run, model, metrics=['accuracy'],
                               validation_data=([X[valid], np.c_[means, scales][valid]], Y[valid]),
                               pretrain_weights=pretrain_weights, **vars(args))
    return X, X_raw, Y, model, args


if __name__ == '__main__':
    X, X_raw, Y, model, args = main()
