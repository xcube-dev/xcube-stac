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


import numpy as np
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonDateSchema,
    JsonNumberSchema,
    JsonStringSchema,
)


DATA_STORE_ID = "stac"

MAP_MIME_TYP_DATAOPENER_ID = {
    "application/netcdf": ("dataset:netcdf:https", "dataset:netcdf:s3"),
    "application/x-netcdf": ("dataset:netcdf:https", "dataset:netcdf:s3"),
    "application/vnd+zarr": ("dataset:zarr:https", "dataset:zarr:s3"),
    "application/zarr": ("dataset:zarr:https", "dataset:zarr:s3"),
    "image/jp2": ("dataset:geotiff:https", "dataset:geotiff:s3"),
    "image/tiff": ("dataset:geotiff:https", "dataset:geotiff:s3"),
}
DATASET_OPENER_ID = []
for vals in MAP_MIME_TYP_DATAOPENER_ID.values():
    for val in vals:
        DATASET_OPENER_ID.append(val)
DATASET_OPENER_ID = tuple(np.unique(DATASET_OPENER_ID))

STAC_SEARCH_PARAMETERS = dict(
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
        title="Bounding box [x1,y1,x2,y2] in geographical coordinates",
    ),
    collections=JsonArraySchema(
        items=(JsonStringSchema(min_length=0)),
        unique_items=True,
        title="Collection IDs",
        description=("Collection IDs to be included in the search request."),
    ),
)

STAC_OPEN_PARAMETERS = dict(
    asset_names=JsonArraySchema(
        items=(JsonStringSchema(min_length=0)),
        unique_items=True,
        title="Names of assets",
        description=("Names of assets which will be included in the data cube."),
    )
)

# Bucket naming rules:
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
AWS_REGEX_BUCKET_NAME = (
    r"(?!^([0-9]{1,3}\.){3}[0-9]{1,3}$)"
    r"(?!(^xn--|^sthree-|^sthree-configurator|"
    r".+--ol-s3$|.+-s3alias$))"
    r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
# Region names: https://docs.aws.amazon.com/general/latest/gr/s3.html
AWS_REGION_NAMES = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-1",
    "af-south-1",
    "ap-east-1",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ca-central-1",
    "eu-central-2",
    "ca-west-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "eu-south-2",
    "eu-north-1",
    "il-central-1",
    "me-south-1",
    "sa-east-1",
    "us-gov-east-1",
    "us-gov-east-1",
]
