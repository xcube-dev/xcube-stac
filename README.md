# This repo is under development. 

# xcube-stac
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/dcs4cop/xcube-smos)](https://github.com/dcs4cop/xcube-smos)


`xcube-stac` is a Python package and 
[xcube plugin](https://xcube.readthedocs.io/en/latest/plugins.html) that adds a 
[data store](https://xcube.readthedocs.io/en/latest/api.html#data-store-framework) 
named `stac` to xcube. The data store is used to 
access data from the [STAC - SpatioTemporal Asset Catalogs](https://stacspec.org/en/).


## Setup

### Installing the xcube-stac plugin from the repository

Installing xcube-cmems directly from the git repository, clone the repository,
direct into `xcube-stac`, and follow the steps below:

```
$ conda env create -f environment.yml
$ conda activate xcube-stac
$ pip install .
```
This installs all the dependencies of `xcube-stac` into a fresh conda environment,
then installs xcube-stac into this environment from the repository.
