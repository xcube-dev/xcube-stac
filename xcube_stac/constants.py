# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
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
DATA_STORE_ID_CDSE_ARDC = "stac-cdse-ardc"
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

# xcube specific constants
DATA_STORE_ID_XCUBE = "stac-xcube"

# other constants
TILE_SIZE = 2048
LOG = logging.getLogger("xcube.stac")
FloatInt = float | int
CONVERSION_FACTOR_DEG_METER = 111320

# constants for data access
PROTOCOLS = ["https", "s3"]
MAP_MIME_TYP_FORMAT = {
    "application/netcdf": "netcdf",
    "application/x-netcdf": "netcdf",
    "application/vnd+zarr": "zarr",
    "application/zarr": "zarr",
    "image/tiff": "geotiff",
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
    "dataset:geotiff:https",
    "mldataset:geotiff:https",
    "dataset:levels:https",
    "mldataset:levels:https",
    "dataset:netcdf:s3",
    "dataset:zarr:s3",
    "dataset:geotiff:s3",
    "mldataset:geotiff:s3",
    "dataset:levels:s3",
    "mldataset:levels:s3",
    "dataset:format:stac-cdse",
)
MLDATASET_FORMATS = ["levels", "geotiff"]

# parameter schemas
SCHEMA_URL = JsonStringSchema(title="URL to STAC catalog")
SCHEMA_S3_STORE = S3FsAccessor.get_storage_options_schema().properties
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
    items=(JsonStringSchema(min_length=1)),
    unique_items=True,
    title="Names of assets",
    description="Names of assets which will be included in the data cube.",
)

SCHEMA_APPLY_SCALING = JsonBooleanSchema(
    title="Apply scaling, offset, and no-data values to data.", default=False
)
SCHEMA_SPATIAL_RES = JsonNumberSchema(title="Spatial Resolution", exclusive_minimum=0.0)
SCHEMA_CRS = JsonStringSchema(title="Coordinate reference system", default="EPSG:4326")
