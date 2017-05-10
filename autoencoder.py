import numpy as np
np.random.seed(0)
from keras.layers import (Input, Dense, TimeDistributed, LSTM, GRU, Dropout, merge,
                          Flatten, RepeatVector, Recurrent, Bidirectional, SimpleRNN)
from keras.models import Model

import sample_data
import keras_util as ku


def encoder(model_input, layer, size, num_layers, drop_frac=0.0, output_size=None,
            bidirectional=False, **parsed_args):
    """Encoder module of autoencoder architecture.

    Can be used either as the encoding component of an autoencoder or as a standalone
    encoder, which takes (possibly irregularly-sampled) time series as inputs and produces
    a fixed-length vector as output.

    model_input: `keras.layers.Input`
        Input layer containing (y) or (dt, y) values
    layer: `keras.layers.Recurrent`
        Desired `keras` recurrent layer class
    size: int
        Number of units within each hidden layer
    num_layers: int
        Number of hidden layers
    drop_frac: float
        Dropout rate
    output_size: int, optional
        Size of encoding layer; defaults to `size`
    bidirectional: bool, optional
        Whether the bidirectional version of `layer` should be used; defaults to `False`
    """

    if output_size is None:
        output_size = size
    encode = model_input
    for i in range(num_layers):
        wrapper = Bidirectional if bidirectional else lambda x: x
        encode = wrapper(layer(size, name='encode_{}'.format(i),
                               return_sequences=(i < num_layers - 1)))(encode)
        if drop_frac > 0.0:
            encode = Dropout(drop_frac, name='drop_encode_{}'.format(i))(encode)
    encode = Dense(output_size, activation='linear', name='encoding')(encode)
    return encode


def decoder(encode, layer, n_step, size, num_layers, drop_frac=0.0, aux_input=None,
            bidirectional=False, **parsed_args):
    """Decoder module of autoencoder architecture.

    Can be used either as the decoding component of an autoencoder or as a standalone
    decoder, which takes a fixed-length input vector and generates a length-`n_step`
    time series as output.

    layer: `keras.layers.Recurrent`
        Desired `keras` recurrent layer class
    n_step: int
        Length of output time series
    size: int
        Number of units within each hidden layer
    num_layers: int
        Number of hidden layers
    drop_frac: float
        Dropout rate
    aux_input: `keras.layers.Input`, optional
        Input layer containing `dt` values; if `None` then the sequence is assumed to be
        evenly-sampled
    bidirectional: bool, optional
        Whether the bidirectional version of `layer` should be used; defaults to `False`
    """
    decode = RepeatVector(n_step, name='repeat')(encode)
    if aux_input is not None:
        decode = merge([aux_input, decode], mode='concat')

    for i in range(num_layers):
        if drop_frac > 0.0 and i > 0:  # skip these for first layer for symmetry
            decode = Dropout(drop_frac, name='drop_decode_{}'.format(i))(decode)
        wrapper = Bidirectional if bidirectional else lambda x: x
        decode = wrapper(layer(size, name='decode_{}'.format(i),
                               return_sequences=True))(decode)

    decode = TimeDistributed(Dense(1, activation='linear'), name='time_dist')(decode)
    return decode


def main(args=None):
    """Generate random periodic time series and train an autoencoder model.
    
    args: dict
        Dictionary of values to override default values in `keras_util.parse_model_args`;
        can also be passed via command line. See `parse_model_args` for full list of
        possible arguments.
    """
    args = ku.parse_model_args(args)

    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    X, Y, X_raw = sample_data.periodic(args.N_train + args.N_test, args.n_min, args.n_max,
                                       even=args.even, noise_sigma=args.sigma,
                                       kind=args.data_type)

    if args.even:
        X = X[:, :, 1:2]
        X_raw = X_raw[:, :, 1:2]
    else:
        X[:, :, 0] = ku.times_to_lags(X_raw[:, :, 0])
        X[np.isnan(X)] = -1.
        X_raw[np.isnan(X_raw)] = -1.

    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN}

    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    if args.even:
        model_input = main_input
        aux_input = None
    else:
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
 
    if args.even:
        history = ku.train_and_log(X[train], X_raw[train], run, model, **vars(args))
    else:
        sample_weight = (X[train, :, -1] != -1)
        history = ku.train_and_log({'main_input': X[train], 'aux_input': X[train, :, 0:1]},
                                   X_raw[train, :, 1:2], run, model,
                                   sample_weight=sample_weight, **vars(args))
    return X, Y, X_raw, model, args


if __name__ == '__main__':
    X, Y, X_raw, model, args = main()
