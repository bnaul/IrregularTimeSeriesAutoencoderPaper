import numpy as np
import joblib
from keras.layers import Input, LSTM, GRU, SimpleRNN
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import keras_util as ku
from autoencoder import encoder, decoder
from light_curve import LightCurve


def preprocess(X_raw, m_max=np.inf):
    X = X_raw.copy()

    wrong_units =  np.all(np.isnan(X[:, :, 1])) | (np.nanmax(X[:, :, 1], axis=1) > m_max)
    X = X[~wrong_units, :, :]

    # Replace times w/ lags
    X[:, :, 0] = ku.times_to_lags(X[:, :, 0])

    means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
    X[:, :, 1] -= means

    scales = np.atleast_2d(np.nanstd(X[:, :, 1], axis=1)).T
    X[:, :, 1] /= scales

    # Drop_errors from input; only used as weights
    X = X[:, :, :2]

    return X, means, scales, wrong_units


def main(args=None):
    """Train an autoencoder model from `LightCurve` objects saved in
    `args.survey_files`.
    
    args: dict
        Dictionary of values to override default values in `keras_util.parse_model_args`;
        can also be passed via command line. See `parse_model_args` for full list of
        possible arguments.
    """
    args = ku.parse_model_args(args)

    np.random.seed(0)

    if not args.survey_files:
        raise ValueError("No survey files given")
    lc_lists = [joblib.load(f) for f in args.survey_files]
    n_reps = [max(len(y) for y in lc_lists) // len(x) for x in lc_lists]
    combined = sum([x * i for x, i in zip(lc_lists, n_reps)], [])
    if args.lomb_score:
        combined = [lc for lc in combined if lc.best_score >= args.lomb_score]
    if args.ss_resid:
        combined = [lc for lc in combined if lc.ss_resid <= args.ss_resid]
    split = [el for lc in combined for el in lc.split(args.n_min, args.n_max)]
    if args.period_fold:
        for lc in split:
            lc.period_fold()
    X_list = [np.c_[lc.times, lc.measurements, lc.errors] for lc in split]

    X_raw = pad_sequences(X_list, value=np.nan, dtype='float', padding='post')
    if args.N_train is not None:
        X_raw = X_raw[:args.N_train]

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN}
    X, means, scales, wrong_units = preprocess(X_raw, args.m_max)
    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
    model_input = [main_input, aux_input]
    encode = encoder(main_input, layer=model_type_dict[args.model_type], 
                     output_size=args.embedding, **vars(args))
    decode = decoder(encode, num_layers=args.decode_layers if args.decode_layers
                                                           else args.num_layers,
                     layer=model_type_dict[args.decode_type if args.decode_type
                                           else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input,
                     **{k: v for k, v in vars(args).items() if k != 'num_layers'})
    model = Model(model_input, decode)

    run = ku.get_run_id(**vars(args))

    errors = X_raw[:, :, 2] / scales
    sample_weight = 1. / errors
    sample_weight[np.isnan(sample_weight)] = 0.0
    X[np.isnan(X)] = 0.

    history = ku.train_and_log({'main_input': X, 'aux_input': np.delete(X, 1, axis=2)},
                               X[:, :, [1]], run, model, sample_weight=sample_weight,
                               errors=errors, validation_split=0.0, **vars(args))

    return X, X_raw, model, means, scales, wrong_units, args


if __name__ == '__main__':
    X, X_raw, model, means, scales, wrong_units, args = main()
