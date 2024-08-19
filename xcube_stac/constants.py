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
    JsonComplexSchema,
    JsonBooleanSchema,
    JsonDateSchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

LOG = logging.getLogger("xcube.stac")
FloatInt = Union[float, int]

COLLECTION_PREFIX = "collections/"

DATA_STORE_ID = "stac"
DATA_STORE_ID_CDSE = "stac-cdse"
DATA_STORE_ID_XCUBE = "stac-xcube"

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

CDSE_MAP_INSTRUEMNT_FORMAT = {"Sentinel-2": ".jp2", "SMOS": ".nc"}

MAP_MIME_TYP_FORMAT = {
    "application/netcdf": "netcdf",
    "application/x-netcdf": "netcdf",
    "application/vnd+zarr": "zarr",
    "application/zarr": "zarr",
    "image/tiff": "geotiff",
}

MAP_CDSE_COLLECTION_FORMAT = {"SMOS": "netcdf"}

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
    "dataset:geotiff:https",
    "mldataset:geotiff:https",
    "dataset:levels:https",
    "mldataset:levels:https",
    "dataset:netcdf:s3",
    "dataset:zarr:s3",
    "dataset:jp2:s3",
    "dataset:geotiff:s3",
    "mldataset:geotiff:s3",
    "dataset:levels:s3",
    "mldataset:levels:s3",
)

MLDATASET_FORMATS = ["levels", "geotiff"]

_STAC_MODE_SCHEMA = JsonComplexSchema(
    one_of=[
        JsonStringSchema(
            title="Backend for stacking STAC items",
            description="So far, only 'odc-stac' is supported as a backend.",
            const="odc-stac",
        ),
        JsonBooleanSchema(
            title="Decide if stacking of STAC items is applied",
            description="If True, 'odc-stac' is used as a default backend.",
            default=False,
        ),
    ],
)

STAC_STORE_PARAMETERS = dict(
    url=JsonStringSchema(title="URL to STAC catalog"), stack_mode=_STAC_MODE_SCHEMA
).update(S3FsAccessor.get_storage_options_schema().properties)


_STAC_SEARCH_ADDITIONAL_QUERY = JsonObjectSchema(
    additional_properties=True,
    title="Additional query options used during item search of STAC API.",
    description=(
        "If STAC Catalog is conform with query extension, "
        "additional filtering based on the properties of Item objects "
        "is supported. For more information see "
        "https://github.com/stac-api-extensions/query"
    ),
)

STAC_SEARCH_PARAMETERS_STACK_MODE = dict(
    time_range=JsonArraySchema(
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
    ),
    bbox=JsonArraySchema(
        items=(
            JsonNumberSchema(),
            JsonNumberSchema(),
            JsonNumberSchema(),
            JsonNumberSchema(),
        ),
        title="Bounding box [x1,y1,x2,y2] in geographical coordinates.",
    ),
)

STAC_SEARCH_PARAMETERS = dict(
    **STAC_SEARCH_PARAMETERS_STACK_MODE,
    collections=JsonArraySchema(
        items=(JsonStringSchema(min_length=0)),
        unique_items=True,
        title="Collection IDs",
        description="Collection IDs to be included in the search request.",
    ),
    query=_STAC_SEARCH_ADDITIONAL_QUERY,
)

STAC_OPEN_PARAMETERS = dict(
    asset_names=JsonArraySchema(
        items=(JsonStringSchema(min_length=0)),
        unique_items=True,
        title="Names of assets",
        description="Names of assets which will be included in the data cube.",
    )
)

STAC_OPEN_PARAMETERS_STACK_MODE = dict(
    time_range=JsonArraySchema(
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
    ),
    bbox=JsonArraySchema(
        items=(
            JsonNumberSchema(),
            JsonNumberSchema(),
            JsonNumberSchema(),
            JsonNumberSchema(),
        ),
        title="Bounding box [x1,y1,x2,y2] in geographical coordinates.",
    ),
    query=_STAC_SEARCH_ADDITIONAL_QUERY,
)
