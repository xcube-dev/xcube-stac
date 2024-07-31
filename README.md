# xcube-stac

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/dcs4cop/xcube-smos)](https://github.com/dcs4cop/xcube-smos)

`xcube-stac` is a Python package and
[xcube plugin](https://xcube.readthedocs.io/en/latest/plugins.html) that adds a
[data store](https://xcube.readthedocs.io/en/latest/api.html#data-store-framework)
named `stac` to xcube. The data store is used to access data from the
[STAC - SpatioTemporal Asset Catalogs](https://stacspec.org/en/).

## Setup

### Installing the xcube-stac plugin from the repository

Installing xcube-stac directly from the git repository, clone the repository,
direct into `xcube-stac`, and follow the steps below:

```bash
conda env create -f environment.yml
conda activate xcube-stac
pip install .
```

This installs all the dependencies of `xcube-stac` into a fresh conda
environment, then installs xcube-stac into this environment from the
repository.

## Testing

To run the unit test suite:

```bash
pytest
```

To analyze test coverage (after installing pytest as above):

```bash
pytest --cov=xcube_stac
```

To produce an HTML
[coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html):

```bash
pytest --cov-report html --cov=xcube_stac
```

### Some notes on the strategy of unittesting

The unit test suite uses [pytest-recording](https://pypi.org/project/pytest-recording/)
to mock STAC catalogs. During development an actual HTTP request is performed
to a STAC catalog and the responses are saved in `cassettes/**.yaml` files.
During testing, only the `cassettes/**.yaml` files are used without an actual
HTTP request. During development run

```bash
pytest -v -s --record-mode new_episodes
```

which saves the responses to `cassettes/**.yaml`. Note that `--record-mode new_episodes`
overwrites all cassettes. If the user only wants to write cassettes which are not saved
already, `--record-mode once` can be used.
[pytest-recording](https://pypi.org/project/pytest-recording/) supports all records modes
given by [VCR.py](https://vcrpy.readthedocs.io/en/latest/usage.html#record-modes).
After recording the cassettes, testing can be then performed as usual.

## Usage

the folder `examples/notebooks` provides example notebooks which gives an overview
of how to use the `stac` data store.  


### Query extension
https://github.com/stac-api-extensions/query

https://github.com/radiantearth/stac-api-spec/tree/release/v1.0.0/item-search#overview