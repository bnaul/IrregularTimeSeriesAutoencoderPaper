# Neural network autoencoders for unevenly sampled time series
[![DOI](https://zenodo.org/badge/90776775.svg)](https://zenodo.org/badge/latestdoi/90776775)

Code accompanying "A recurrent neural network for classification of unevenly sampled variable stars".

- Code for scores/figures is found in `figures.ipynb`
- Autoencoder network architecture is defined in `autoencoder.py`
- Experiments for simulated data are found in `autoencoder.main`
- Experiments for light curve data are found in `survey_autoencoder.main`
    - Light curve data is in `./data`
- Model weights are saved in `./keras_logs`

See mirror at https://gitlab.com/cesium-ml/IrregularTimeSeriesAutoencoderPaper to download the full set of input data.
