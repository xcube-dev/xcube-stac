# xcube-stac

[![Build Status](https://github.com/xcube-dev/xcube-stac/actions/workflows/unittest-workflow.yml/badge.svg?branch=main)](https://github.com/xcube-dev/xcube-smos/actions)
[![codecov](https://codecov.io/gh/xcube-dev/xcube-stack/branch/main/graph/badge.svg)](https://app.codecov.io/gh/xcube-dev/xcube-stac)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/dcs4cop/xcube-smos)](https://github.com/xcube-dev/xcube-stac/blob/main/LICENSE)

`xcube-stac` is a Python package and
[xcube plugin](https://xcube.readthedocs.io/en/latest/plugins.html) that adds a
[data store](https://xcube.readthedocs.io/en/latest/api.html#data-store-framework)
named `stac` to xcube. The data store is used to access data from the
[STAC - SpatioTemporal Asset Catalogs](https://stacspec.org/en/).

## Table of contents
1. [Setup](#setup)
   1. [Installing the xcube-stac plugin from the repository](#install_source)
2. [Overview](#overview)
   1. [General structure of a STAC catalog](#stac_catalog)
   2. [General functionality of xcube-stac](#func_xcube_stac)
3. [Introduction to xcube-stac](#intro_xcube_stac)
   1. [Overview of Jupyter notebooks](#overview_notebooks)
   2. [Getting started](#getting_started)
4. [Testing](#testing)
   1. [Some notes on the strategy of unit-testing](#unittest_strategy)

## Setup <a name="setup"></a>

### Installing the xcube-stac plugin from the repository <a name="install_source"></a>

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

## Overview <a name="overview"></a>

### General structure of a STAC catalog <a name="stac_catalog"></a>
A SpatioTemporal Asset Catalog (STAC) consists of three main components: catalog,
collection, and item. Each item can contain multiple assets, each linked to a data
source. Items are associated with a timestamp or temporal range and a bounding box
describing the spatial extent of the data. 

Items within a collection generally exhibit
similarities. For example, a STAC catalog might contain multiple collections
corresponding to different space-borne instruments. Each item represents a measurement
covering a specific spatial area at a particular timestamp. For a multi-spectral
instrument, different bands can be stored as separate assets.

A STAC catalog can comply with the [STAC API - Item Search](https://github.com/radiantearth/stac-api-spec/tree/release/v1.0.0/item-search#stac-api---item-search)
conformance class, enabling server-side searches for items based on specific
parameters. If this compliance is not met, only client-side searches are possible,
which can be slow for large STAC catalogs.

### General functionality of xcube-stac <a name="func_xcube_stac"></a>
The xcube-stac plugin reads the data sources from the STAC catalog and opens the data
in an analysis ready form following the [xcube dataset convetion](https://xcube.readthedocs.io/en/latest/cubespec.html).
By default, a data ID represents one item, which is opened as a dataset, with each
asset becoming a data variable within the dataset. 

Additionally, a stack mode is
available, enabling the stacking of items using [odc-stac](https://odc-stac.readthedocs.io/en/latest/).
This allows for mosaicking multiple tiles and concatenating the datacube along the
temporal axis.

Also, [stackstac](https://stackstac.readthedocs.io/en/latest/) has been
considered during the evaluation of python libraries supporting stacking of STAC items.
However, the [benchmarking report](https://benchmark-odc-stac-vs-stackstac.netlify.app/)
comparing stackstac and odc-stac shows that ocd-stac outperforms stackstac. Furthermore,
stackstac shows an [issue](https://github.com/gjoseph92/stackstac/issues/196) in making
use of the overview levels of COGs files. Still, stackstac shows high popularity in the
community and might be supported in the future. 

## Introduction to xcube-stac <a name="intro_xcube_stac"></a> 

### Overview of Jupyter notebooks <a name="overview_notebooks"></a> 
The following Jupyter notebooks provide some examples: 

* `example/notebooks/earth_search_sentinel2_l2a_stack_mode.ipynb`:
  This notebook shows an example how to stack multiple tiles of Sentinel-2 L2A data
  from Earth Search by Element 84 STAC API. It shows stacking of individual tiles and
  mosaicking of multiple tiles measured on the same solar day.
* `example/notebooks/geotiff_nonsearchable_catalog.ipynb`:
  This notebook shows an example how to load a GeoTIFF file from a non-searchable
  STAC catalog.
* `example/notebooks/geotiff_searchable_catalog.ipynb`:
  This notebook shows an example how to load a GeoTIFF file from a searchable
  STAC catalog.
* `example/notebooks/netcdf_searchable_catalog.ipynb`:
  This notebook shows an example how to load a NetCDF file from a searchable
  STAC catalog.
* `example/notebooks/xcube_server_stac_s3.ipynb`:
  This notebook shows an example how to open data sources published by xcube server
  via the STAC API.

### Getting started <a name="getting_started"></a> 

The xcube [data store framework](https://xcube.readthedocs.io/en/latest/dataaccess.html#data-store-framework)
allows to easily access data in an analysis ready format, following the few lines of
code below. 

```python
from xcube.core.store import new_data_store

store = new_data_store(
    "stac",
    url="https://earth-search.aws.element84.com/v1"
)
ds = store.open_data(
    "collections/sentinel-2-l2a/items/S2B_32TNT_20200705_0_L2A",
    data_type="dataset"
)
```
The data ID `"collections/sentinel-2-l2a/items/S2B_32TNT_20200705_0_L2A"` points to the
[STAC item's JSON](https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md)
and is specified by the segment of the URL that follows the catalog's URL. The
`data_type` can be set to `dataset` and `mldataset`, which returns a `xr.Dataset` and
a [xcube multi-resoltuion dataset](https://xcube.readthedocs.io/en/latest/mldatasets.html),
respectively. Note that in the above example, if `data_type` is not assigned,
a multi-resolution dataset will be returned. This is because the item's asset links to
GeoTIFFs, which are opened as multi-resolution datasets by default.

To use the stac-mode, initiate a stac store with the argument `stack_mode=True`.

```python
from xcube.core.store import new_data_store

store = new_data_store(
    "stac",
    url="https://earth-search.aws.element84.com/v1",
    stack_mode=True
)
ds = store.open_data(
    "sentinel-2-l2a",
    data_type="dataset",
    bbox=[9.1, 53.1, 10.7, 54],
    time_range= ["2020-07-01", "2020-08-01"],
    query={"s2:processing_baseline": {"eq": "02.14"}},
)
```

In the stacking mode, the data IDs are the collection IDs within the STAC catalog. To
get Sentinel-2 L2A data, we assign `data_id` to `"sentinel-2-l2a"`. The bounding box and
time range are assigned to define the temporal and spatial extent of the data cube. 
Additionally, for this example, we need to set a query argument to select a specific
[Sentinel-2 processing baseline](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-L2Aprocessingbaseline),
as the collection contains multiple items for the same tile with different processing
procedures. Note that this requirement can vary between collections and must be
specified by the user. To set query arguments, the STAC catalog needs to be conform with
the [query extension](https://github.com/stac-api-extensions/query).

The stacking is performed using [odc-stac](https://odc-stac.readthedocs.io/en/latest/).
All arguments of [odc.stac.load](https://odc-stac.readthedocs.io/en/latest/_api/odc.stac.load.html)
can be passed into the `open_data(...)` method, which forwards them to the
`odc.stac.load` function.

To apply mosaicking, we need to assign `groupby="solar_day"`, as shown in the
[documentation of `odc.stac.load`](https://odc-stac.readthedocs.io/en/latest/_api/odc.stac.load.html).
The following few lines of code show a small example including mosaicking.  

```python
from xcube.core.store import new_data_store

store = new_data_store(
    "stac",
    url="https://earth-search.aws.element84.com/v1",
    stack_mode=True
)
ds = store.open_data(
    "sentinel-2-l2a",
    data_type="dataset",
    bbox=[9.1, 53.1, 10.7, 54],
    time_range= ["2020-07-01", "2020-08-01"],
    query={"s2:processing_baseline": {"eq": "02.14"}},
    groupby="solar_day",
)
```

## Testing <a name="testing"></a>

To run the unit test suite:

```bash
pytest
```

To analyze test coverage:

```bash
pytest --cov=xcube_stac
```

To produce an HTML
[coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html):

```bash
pytest --cov-report html --cov=xcube_stac
```

### Some notes on the strategy of unit-testing <a name="unittest_strategy"></a>

The unit test suite uses [pytest-recording](https://pypi.org/project/pytest-recording/)
to mock STAC catalogs. During development an actual HTTP request is performed
to a STAC catalog and the responses are saved in `cassettes/**.yaml` files.
During testing, only the `cassettes/**.yaml` files are used without an actual
HTTP request. During development, to save the responses to `cassettes/**.yaml`, run

```bash
pytest -v -s --record-mode new_episodes
```
Note that `--record-mode new_episodes` overwrites all cassettes. If the user only
wants to write cassettes which are not saved already, `--record-mode once` can be used.
[pytest-recording](https://pypi.org/project/pytest-recording/) supports all records modes given by [VCR.py](https://vcrpy.readthedocs.io/en/latest/usage.html#record-modes).
After recording the cassettes, testing can be then performed as usual.