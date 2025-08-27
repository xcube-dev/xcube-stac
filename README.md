# xcube-stac

[![Build Status](https://github.com/xcube-dev/xcube-stac/actions/workflows/unittest-workflow.yml/badge.svg?branch=main)](https://github.com/xcube-dev/xcube-stac/actions)
[![codecov](https://codecov.io/gh/xcube-dev/xcube-stac/graph/badge.svg?token=ktcp1maEgz)](https://codecov.io/gh/xcube-dev/xcube-stac)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/dcs4cop/xcube-smos)](https://github.com/xcube-dev/xcube-stac/blob/main/LICENSE)

`xcube-stac` is a Python package and a [xcube plugin](https://xcube.readthedocs.io/en/latest/plugins.html)
that provides a [data store](https://xcube.readthedocs.io/en/latest/api.html#data-store-framework)
for accessing data from [STAC (SpatioTemporal Asset Catalogs)](https://stacspec.org/en/).


## Table of contents
1. [Overview](#overview)
   1. [General structure of a STAC catalog](#general-structure-of-a-stac-catalog)
   2. [General functionality of xcube-stac](#general-functionality-of-xcube-stac)
   3. [Special support for the CDSE STAC API](#special-support-for-the-cdse-stac-api)
2. [Setup](#setup)
   1. [Installing the xcube-stac plugin](#installing-the-xcube-stac-plugin)
   2. [Getting S3 credentials for CDSE data access ](#getting-s3-credentials-for-cdse-data-access)
3. [Introduction to xcube-stac](#introduction-to-xcube-stac)
   1. [Overview of Jupyter notebooks](#overview-of-jupyter-notebooks)
   2. [Getting started](#getting-started)
4. [Testing](#testing)
   1. [Some notes on the strategy of unit-testing](#some-notes-on-the-strategy-of-unit-testing)

--- 

## Overview

### General structure of a STAC catalog

A **SpatioTemporal Asset Catalog (STAC)** typically consists of three main components:

- **Catalogs**
- **Collections**
- **Items**

Each **item** represents a spatiotemporal observation and includes:

- a timestamp or temporal range
- a bounding box defining its spatial extent
- one or more **assets**, each linking to a data source (such as imagery or metadata)

Items within a **collection** generally share common characteristics. For example, 
a STAC catalog might have separate collections for different satellite data products. 
Each item would then correspond to a specific measurement covering a certain area at a 
particular time. In multi-spectral instruments, different bands are often stored as 
individual assets.

Most STAC catalogs conform to the [STAC API - Item Search](https://github.com/radiantearth/stac-api-spec/tree/release/v1.0.0/item-search#stac-api---item-search) 
specification, enabling efficient **server-side queries** based on spatial, temporal, 
or attribute filters. Without this conformance, only **client-side searches** are 
possible, which can be slow for large catalogs.

---

### General functionality of xcube-stac

The **xcube-stac plugin** reads data sources described by a STAC catalog and opens 
them as `xr.Dataset` that follows the [xcube dataset convention](https://xcube.readthedocs.io/en/latest/cubespec.html).

By default:

- A **data ID** corresponds to a single STAC item.
- Each item is opened as a dataset, with each asset becoming a data variable within 
  that dataset.
- In ARDC-mode data stores (`*-ardc`), a data ID can also correspond to a 
  **collection ID**.


#### Available data stores

1. `"stac"`: General STAC data store. Uses [xcube's file-system data stores](https://xcube.readthedocs.io/en/latest/dataaccess.html#filesystem-based-data-stores) to access Zarr, NetCDF, or GeoTIFF sources.
2. `"stac-xcube"`: Accesses datasets published by the [xcube server STAC API](https://xcube.readthedocs.io/en/latest/webapi.html).
3. `"stac-cdse"`: Tailored for the [CDSE STAC API](https://browser.stac.dataspace.copernicus.eu/?.language=en).
   specific support is provided for the collections listed in [Special support for the CDSE STAC API](#special-support-for-the-cdse-stac-api).
4. `"stac-cdse-ardc"`: Generates **3D spatiotemporal analysis-ready data cubes (ARDCs)** 
   from multiple STAC items for the supported CDSE collections listed in [Special support for the CDSE STAC API](#special-support-for-the-cdse-stac-api).
5. `"stac-pc"`: Tailored for the [Planetary Computer STAC API](https://planetarycomputer.microsoft.com/).
   Specific support is provided for the collections listed in [Special support for the Planetary Computer STAC API](#special-support-for-the-planetary-computer-stac-api).
6. `"stac-pc-ardc"`: Generates ARDCs from multiple STAC items for the supported
   Planetary Computer collections listed in [Special support for the Planetary Computer STAC API](#special-support-for-the-planetary-computer-stac-api).

---

### Building analysis-ready data stores (ARDCs)

Some STAC catalogs are designed to enable the creation of 
**analysis-ready data cubes (ARDCs)** from multiple STAC items in a collection.  


Currently, ARDC support is provided for:

- [CDSE STAC API](#special-support-for-the-cdse-stac-api)
- [Planetary Computer STAC API](#special-support-for-the-planetary-computer-stac-api)

The workflow for building a 3D analysis-ready cube includes:

1. **Querying** products from the CDSE STAC API for a specified time range and spatial 
   extent.
2. **Retrieving** observations using a lazy-loading reader. (Different collections use 
   different readers depending on the underlying data format.)
3. **Mosaicking** spatial tiles into single images per timestamp.
4. **Stacking** these mosaics along the temporal axis to produce a 3D data cube.

> **Note:**  
> During evaluation, we also considered [odc-stac](https://odc-stac.readthedocs.io/en/latest/) 
> and [stackstac](https://stackstac.readthedocs.io/en/latest/) for stacking STAC items.  
> However, both libraries rely on `rasterio.open` (GDAL drivers) to read data, which 
> prevents accessing data directly from the [CDSE S3 endpoint](https://documentation.dataspace.copernicus.eu/APIs/S3.html) 
> due to blocked AWS environments.
>
> Among them, a [benchmark](https://benchmark-odc-stac-vs-stackstac.netlify.app/) shows 
> that `odc-stac` outperforms `stackstac`. Additionally, `stackstac` is less mature, 
> with known issues (e.g., [#196](https://github.com/gjoseph92/stackstac/issues/196)) 
> handling COG overviews. Despite this, both are widely used in the community and may
> be supported in future releases.


#### Special support for the [CDSE STAC API](https://browser.stac.dataspace.copernicus.eu/?language=en)

Currently, we support the following collections and data IDs:

- [`sentinel-2-l1c`](https://browser.stac.dataspace.copernicus.eu/collections/sentinel-2-l1c)
- [`sentinel-2-l2a`](https://browser.stac.dataspace.copernicus.eu/collections/sentinel-2-l2a)
- [`sentinel-3-syn-2-syn-ntc`](https://browser.stac.dataspace.copernicus.eu/collections/sentinel-3-syn-2-syn-ntc)


#### Special support for the [Planetary Computer STAC API](https://planetarycomputer.microsoft.com/)

Currently, we support the following collections and data IDs:

- [`sentinel-2-l2a`](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)

---

## Setup

### Installing the xcube-stac plugin

This section describes three alternative methods you can use to install the
xcube-stac plugin.

For installation of conda packages, we recommend
[mamba](https://mamba.readthedocs.io/). It is also possible to use conda,
but note that installation may be significantly slower with conda than with
mamba. If using conda rather than mamba, replace the `mamba` command with
`conda` in the installation commands given below.

#### Installation into a new environment with mamba

This method creates a new environment and installs the latest conda-forge
release of xcube-stac, along with all its required dependencies, into the
newly created environment.

To do so, execute the following commands:

```bash
mamba create --name xcube-stac --channel conda-forge xcube-stac
mamba activate xcube-stac
```
The name of the environment may be freely chosen.

#### Installation into an existing environment with mamba

This method assumes that you have an existing environment, and you want
to install xcube-stac into it.

With the existing environment activated, execute this command:

```bash
mamba install --channel conda-forge xcube-stac
```

Once again, xcube and any other necessary dependencies will be installed
automatically if they are not already installed.

#### Installation into an existing environment from the repository

If you want to install xcube-stac directly from the git repository (for example
in order to use an unreleased version or to modify the code), you can
do so as follows:

```bash
mamba create --name xcube-stac --channel conda-forge --only-deps xcube-stac
mamba activate xcube-stac
git clone https://github.com/xcube-dev/xcube-stac.git
python -m pip install --no-deps --editable xcube-stac/
```

This installs all the dependencies of xcube-stac into a fresh conda environment,
then installs xcube-stac into this environment from the repository.

### Getting S3 credentials for CDSE data access
Note, this step is only needed, if the [CDSE STAC API](https://browser.stac.dataspace.copernicus.eu/?.language=en)
wants to be used. In order to access [EO data via S3 from CDSE](https://documentation.dataspace.copernicus.eu/APIs/S3.html)
one needs to [generate S3 credentials](https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets),
which are required to initiate a `"stac-cdse"` data store. So far, only Sentinel-2 L2A
is supported. An example is shown in a [notebook](examples/notebooks/cdse_senitnel_2.ipynb).


## Introduction to xcube-stac

### Overview of Jupyter notebooks
The following Jupyter notebooks provide some examples: 

* `example/notebooks/geotiff_nonsearchable_catalog.ipynb`:
  This notebook shows an example how to load a GeoTIFF file from a non-searchable
  STAC catalog.
* `example/notebooks/geotiff_searchable_catalog.ipynb`:
  This notebook shows an example how to load a GeoTIFF file from a searchable
  STAC catalog.
* `example/notebooks/netcdf_searchable_catalog.ipynb`:
  This notebook shows an example of how to load a NetCDF file from a searchable
  STAC catalog.
* `example/notebooks/sentinel_2_cdse.ipynb`:
  This notebook shows an example of how to access Sentinel-2 L1C and L2A data using the 
  [CDSE STAC API](https://documentation.dataspace.copernicus.eu/APIs/STAC.html). It shows how to access individual observation tiles and how 
  to generate spatiotemporal 3d analysis-ready data cubes from multiple STAC items.  
* `example/notebooks/sentinel_2_planetary_computer.ipynb`:
  This notebook shows an example of how to access Sentinel-2 L2A data using the 
  [Planetary Computer STAC API](planetarycomputer.microsoft.com). It shows how to 
  access individual observation tiles and how to generate spatiotemporal 3d 
  analysis-ready data cubes from multiple STAC items.  
* `example/notebooks/sentinel_3_cdse.ipynb`: This notebook shows an example of how to 
  access Sentinel-3 Synergy Level-2 Land Surface Reflectance and Aerosol product 
  using the [CDSE STAC API](https://documentation.dataspace.copernicus.eu/APIs/STAC.html). It shows how to access individual observation tiles 
  and how to generate spatiotemporal 3d analysis-ready data cubes from multiple STAC 
  items. 
* `example/notebooks/xcube_server_stac_s3.ipynb`:
  This notebook shows an example of how to open data sources published by xcube server
  via the STAC API.

### Getting started

The xcube [data store framework](https://xcube.readthedocs.io/en/latest/dataaccess.html#data-store-framework)
allows to access data, following the few lines of code below. In the following examples 
[S3 credentials](#getting-s3-credentials-for-cdse-data-access) for CDSE data access is needed

```python
from xcube.core.store import new_data_store

credentials = {
    "key": "xxx",
    "secret": "xxx",
}

store = new_data_store("stac-cdse", **credentials)
ds = store.open_data(
    "collections/sentinel-2-l2a/items/S2B_MSIL2A_20200705T101559_N0500_R065_T32TMT_20230530T175912"
)
ds
```
The data ID `"collections/sentinel-2-l2a/items/S2B_MSIL2A_20200705T101559_N0500_R065_T32TMT_20230530T175912"` 
points to the [CDSE STAC item's JSON](https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20200705T101559_N0500_R065_T32TMT_20230530T175912)
and is specified by the segment of the URL that follows the catalog's URL.

To generate a 3D spatiotemporal datacubes, execute the following lines of code.  

```python
from xcube.core.store import new_data_store

credentials = {
    "key": "xxx",
    "secret": "xxx",
}

store = new_data_store("stac-cdse-ardc", **credentials)
ds = store.open_data(
    data_id="sentinel-2-l2a",
    bbox=[9.7, 53.3, 10.3, 53.8],
    time_range=["2020-07-15", "2020-08-01"],
    spatial_res=10 / 111320, # meter in degree
    crs="EPSG:4326",
    asset_names=["B02", "B03", "B04"],
)
```

In the `stac-cdse-ardc` data store, the data IDs are the collection IDs within the STAC 
catalog. To get Sentinel-2 L2A data, we assign `data_id` to `"sentinel-2-l2a"` in the 
above example. The bounding box and time range are assigned to define the temporal and 
spatial extent of the data cube. The parameter `crs` and `spatial_res` are required as 
well and define the coordinate reference system (CRS) and the spatial resolution, 
respectively. Note, that the bounding box and spatial resolution needs to be given 
in the respective CRS.

## Testing

The test suite uses [pytest-recording](https://pypi.org/project/pytest-recording/)
to mock STAC catalogs. To run the test suite, `pytest` and `pytest-recording` need to
be installed. Then, the test suite can be executed as usual by typing:

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

### Some notes on the strategy of unit-testing

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
After recording the cassettes, testing can be performed as usual.