# Neural network autoencoders for unevenly sampled time series
[![DOI](https://zenodo.org/badge/90776775.svg)](https://zenodo.org/badge/latestdoi/90776775)

Code accompanying "A recurrent neural network for classification of unevenly sampled variable stars".

- Code for scores/figures is found in `figures.ipynb`
- Autoencoder network architecture is defined in `autoencoder.py`
- Experiments for simulated data are found in `autoencoder.main`
- Experiments for light curve data are found in `survey_autoencoder.main`
    - Light curve data is in `./data`
- Model weights are saved in `./keras_logs`

You can cite this code as:
```
@misc{brett_naul_2017_1045560,
  author       = {Brett Naul and
                  Joshua S. Bloom and
                  Fernando Pérez and
                  Stéfan van der Walt},
  title        = {{Code/Data from: "A recurrent neural network for 
                   classification of unevenly sampled variable stars"}},
  month        = nov,
  year         = 2017,
  doi          = {10.5281/zenodo.1045560},
  url          = {https://doi.org/10.5281/zenodo.1045560}
}
```
