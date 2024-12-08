# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from typing import Union

from xcube.core.store.fs.impl.fs import S3FsAccessor
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonDateSchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

# general stac constants
DATA_STORE_ID = "stac"

# cdse specific constants
DATA_STORE_ID_CDSE = "stac-cdse"
CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
MAP_CDSE_COLLECTION_FORMAT = {"Sentinel-2": "jp2"}

# xcube specific constants
DATA_STORE_ID_XCUBE = "stac-xcube"

# general constants for data format
MAP_MIME_TYP_FORMAT = {
    "application/netcdf": "netcdf",
    "application/x-netcdf": "netcdf",
    "application/vnd+zarr": "zarr",
    "application/zarr": "zarr",
    "image/tiff": "geotiff",
    "image/jp2": "jp2",
}
MAP_FILE_EXTENSION_FORMAT = {
    ".nc": "netcdf",
    ".zarr": "zarr",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".geotiff": "geotiff",
    ".levels": "levels",
}
DATA_OPENER_IDS = (
    "dataset:netcdf:https",
    "dataset:zarr:https",
    "dataset:jp2:https",
    "mldataset:jp2:https",
    "dataset:geotiff:https",
    "mldataset:geotiff:https",
    "dataset:levels:https",
    "mldataset:levels:https",
    "dataset:netcdf:s3",
    "dataset:zarr:s3",
    "dataset:jp2:s3",
    "mldataset:jp2:s3",
    "dataset:geotiff:s3",
    "mldataset:geotiff:s3",
    "dataset:levels:s3",
    "mldataset:levels:s3",
)
MLDATASET_FORMATS = ["levels", "geotiff", "jp2"]

# other constants
COLLECTION_PREFIX = "collections/"
STAC_CRS = "EPSG:4326"
TILE_SIZE = 1024
LOG = logging.getLogger("xcube.stac")
FloatInt = Union[float, int]
PROCESSING_BASELINE_KEYS = ["processorVersion", "s2:processing_baseline"]

# parameter schemas
STAC_STORE_PARAMETERS = dict(
    url=JsonStringSchema(title="URL to STAC catalog"),
    stack_mode=JsonBooleanSchema(
        title="Decide if stacking of STAC items is applied",
        description="If True, 'odc-stac' is used as a default backend.",
        default=False,
    ),
)
STAC_STORE_PARAMETERS.update(S3FsAccessor.get_storage_options_schema().properties)

SCHEMA_ADDITIONAL_QUERY = JsonObjectSchema(
    additional_properties=True,
    title="Additional query options used during item search of STAC API.",
    description=(
        "If STAC Catalog is conform with query extension, "
        "additional filtering based on the properties of Item objects "
        "is supported. For more information see "
        "https://github.com/stac-api-extensions/query"
    ),
)
SCHEMA_PROCESSING_LEVEL = JsonStringSchema(
    title="Processing level of Sentinel-2 data", enum=["L1C", "L2A"], default="L2A"
)
SCHEMA_BBOX = JsonArraySchema(
    items=(
        JsonNumberSchema(),
        JsonNumberSchema(),
        JsonNumberSchema(),
        JsonNumberSchema(),
    ),
    title="Bounding box [x1,y1,x2,y2] in geographical coordinates.",
)
SCHEMA_TIME_RANGE = JsonArraySchema(
    items=[
        JsonDateSchema(nullable=True),
        JsonDateSchema(nullable=True),
    ],
    title="Time Range",
    description=(
        "Time range given as pair of start and stop dates. "
        "Dates must be given using format 'YYYY-MM-DD'. "
        "Start and stop are inclusive."
    ),
)
SCHEMA_COLLECTIONS = JsonArraySchema(
    items=(JsonStringSchema(min_length=0)),
    unique_items=True,
    title="Collection IDs",
    description="Collection IDs to be included in the search request.",
)
SCHEMA_ASSET_NAMES = JsonArraySchema(
    items=(JsonStringSchema(min_length=0)),
    unique_items=True,
    title="Names of assets",
    description="Names of assets (bands) which will be included in the data cube.",
)
SCHEMA_SPATIAL_RES = JsonNumberSchema(title="Spatial Resolution", exclusive_minimum=0.0)
SCHEMA_CRS = JsonStringSchema(title="Coordinate reference system", default="EPSG:4326")

STAC_SEARCH_PARAMETERS_STACK_MODE = dict(
    time_range=SCHEMA_TIME_RANGE,
    bbox=SCHEMA_BBOX,
)

STAC_SEARCH_PARAMETERS = dict(
    **STAC_SEARCH_PARAMETERS_STACK_MODE,
    collections=SCHEMA_COLLECTIONS,
    query=SCHEMA_ADDITIONAL_QUERY,
)

STAC_OPEN_PARAMETERS = dict(
    asset_names=SCHEMA_ASSET_NAMES,
    apply_scaling=JsonBooleanSchema(
        title="Apply scaling, offset, and no-data values to data"
    ),
)

STAC_OPEN_PARAMETERS_STACK_MODE = dict(
    **STAC_OPEN_PARAMETERS,
    time_range=SCHEMA_TIME_RANGE,
    bbox=SCHEMA_BBOX,
    crs=SCHEMA_CRS,
    spatial_res=SCHEMA_SPATIAL_RES,
    query=SCHEMA_ADDITIONAL_QUERY,
)
