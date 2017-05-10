import numpy as np
from keras.preprocessing.sequence import pad_sequences


def phase_to_sin_cos(Y):
    """Reparametrize sinusoid parameters:
        w, A, phi, b --> p, A_cos, A_sin, b

    Estimating these parameters seems to be easier in practice.
    """
    w, A, phi, b = Y.T

    A_cos = A * np.sin(phi)
    A_sin = A * np.cos(phi)
    p = w ** -1

    return np.c_[p, A_cos, A_sin, b]


def _random_times(N, even=True, t_max=4 * np.pi, n_min=None, n_max=None, t_shape=2, t_scale=0.05):
    if n_min is None and n_max is None:
        raise ValueError("Either n_min or n_max is required.")
    elif n_min is None:
        n_min = n_max
    elif n_max is None:
        n_max = n_min

    if even:
        return np.tile(np.linspace(0., t_max, n_max), (N, 1))
    else:
        lags = [t_scale * np.random.pareto(t_shape, size=np.random.randint(n_min, n_max + 1))
                for i in range(N)]
        return [np.r_[0, np.cumsum(lags_i)] for lags_i in lags]


def _periodic_params(N, A_min, A_max, w_min, w_max):
    w = 1. / np.random.uniform(1. / w_max, 1. / w_min, size=N)
    A = np.random.uniform(A_min, A_max, size=N)
    phi = 2 * np.pi * np.random.random(size=N)
    b = np.random.normal(scale=1, size=N)

    return w, A, phi, b


def _sinusoid(w, A, phi, b):
    return lambda t: A * np.sin(2 * np.pi * w * t + phi) + b


def periodic(N, n_min, n_max, t_max=4 * np.pi, even=True, A_min=0.5, A_max=2.0,
             noise_sigma=0., w_min=0.1, w_max=1., t_shape=2, t_scale=0.05,
             kind='sinusoid'):
    """Returns periodic data (values, (freq, amplitude, phase, offset))"""
    t = _random_times(N, even, t_max, n_min, n_max, t_shape, t_scale)
    w, A, phi, b = _periodic_params(N, A_min, A_max, w_min, w_max)

    X_list = [np.c_[t[i], _sinusoid(w[i], A[i], phi[i], b[i])(t[i])] for i in range(N)]
    X_raw = pad_sequences(X_list, maxlen=n_max, value=np.nan, dtype='float', padding='post')
    X = X_raw.copy()
    X[:, :, 1] = X_raw[:, :, 1] + np.random.normal(scale=noise_sigma + 1e-9, size=(N, n_max))
    Y = np.c_[w, A, phi, b]
    
    return X, Y, X_raw
