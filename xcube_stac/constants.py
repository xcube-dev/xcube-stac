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
from xcube.util.jsonschema import (JsonArraySchema, JsonBooleanSchema,
                                   JsonDateSchema, JsonIntegerSchema,
                                   JsonNumberSchema, JsonObjectSchema,
                                   JsonStringSchema)

# general stac constants
DATA_STORE_ID = "stac"

# cdse specific constants
DATA_STORE_ID_CDSE = "stac-cdse"
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

# xcube specific constants
DATA_STORE_ID_XCUBE = "stac-xcube"

# other constants
COLLECTION_PREFIX = "collections/"
STAC_CRS = "EPSG:4326"
TILE_SIZE = 1024
LOG = logging.getLogger("xcube.stac")
FloatInt = Union[float, int]

# constants for data access
PROTOCOLS = ["https", "s3"]
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
    ".jp2": "jp2",
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
    description="Names of assets which will be included in the data cube.",
)
SCHEMA_ANGLES_SENTINEL2 = JsonBooleanSchema(
    title="Add viewing and solar angles from Sentinel2 metadata.",
    description=(
        "Viewing and solar angles will be extracted for all spectral "
        "bands defined in keyword `asset_name`."
    ),
    default=False,
)
SCHEMA_APPLY_SCALING = JsonBooleanSchema(
    title="Apply scaling, offset, and no-data values to data."
)
SCHEMA_SPATIAL_RES = JsonNumberSchema(title="Spatial Resolution", exclusive_minimum=0.0)
SCHEMA_CRS = JsonStringSchema(title="Coordinate reference system", default="EPSG:4326")
SCHEMA_TILE_SIZE = JsonArraySchema(
    nullable=True,
    title="Tile size of returned dataset",
    description=(
        "Tile size in y and x (or lat and lon if crs is geographic) "
        "of returned dataset"
    ),
    items=[JsonIntegerSchema(minimum=1), JsonIntegerSchema(minimum=1)],
    default=(1024, 1024),
)
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
    angles_sentinel2=SCHEMA_ANGLES_SENTINEL2,
    apply_scaling=SCHEMA_APPLY_SCALING,
)
STAC_OPEN_PARAMETERS_STACK_MODE = dict(
    **STAC_OPEN_PARAMETERS,
    time_range=SCHEMA_TIME_RANGE,
    bbox=SCHEMA_BBOX,
    crs=SCHEMA_CRS,
    tile_size=SCHEMA_TILE_SIZE,
    spatial_res=SCHEMA_SPATIAL_RES,
    query=SCHEMA_ADDITIONAL_QUERY,
)
